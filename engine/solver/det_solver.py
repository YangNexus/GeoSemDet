"""   
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 D-FINE authors. All Rights Reserved.
""" 

import time  
import json
import datetime
import copy  
import gc

import torch
 
from ..misc import dist_utils, stats, get_weight_size

from ._solver import BaseSolver, ModelSaverFunc   
from .det_engine import train_one_epoch, evaluate
from .baseline_log_monitor import BaselineLogMonitor     
from ..optim.lr_scheduler import FlatCosineLRScheduler, MetricLrReducer  
from ..logger_module import get_logger     
     
RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
logger = get_logger(__name__)
coco_name_list = ['ap', 'ap50', 'ap75', 'aps', 'apm', 'apl', 'ar', 'ar50', 'ar75', 'ars', 'arm', 'arl']  
coco_name_tiny_list = ['ap', 'ap50', 'ap75', 'apt', 'aps', 'apm', 'apl', 'ar', 'ar50', 'ar75', 'art', 'ars', 'arm', 'arl']    
coco_name_vtiny_list = ['ap', 'ap50', 'ap75', 'apvt', 'apt', 'aps', 'apm', 'apl', 'ar', 'ar50', 'ar75', 'arvt', 'art', 'ars', 'arm', 'arl']
     
class DetSolver(BaseSolver):
    def __init__(self, cfg):
        super().__init__(cfg)     
        self.best_stat = self._default_best_stat()     
        self.metric_lr_reducer = None
        self._pending_metric_lr_reducer_state = None  
        self.baseline_log_monitor = None     
  
    @staticmethod
    def _default_best_stat():    
        return {'epoch': -1, }

    def _ensure_best_stat(self):     
        if not isinstance(getattr(self, 'best_stat', None), dict):
            self.best_stat = self._default_best_stat()   

    def state_dict(self):    
        state = super().state_dict()
        self._ensure_best_stat()
        state['best_stat'] = copy.deepcopy(self.best_stat)
        if self.metric_lr_reducer is not None:
            state['metric_lr_reducer_state'] = self.metric_lr_reducer.state_dict()
        return state     
   
    def load_state_dict(self, state):
        super().load_state_dict(state) 
        loaded_best_stat = state.get('best_stat', None)
        if isinstance(loaded_best_stat, dict):
            self.best_stat = copy.deepcopy(loaded_best_stat) 
        else: 
            self.best_stat = self._default_best_stat()
        if 'metric_lr_reducer_state' in state:   
            reducer_state = copy.deepcopy(state['metric_lr_reducer_state'])
            if self.metric_lr_reducer is not None:
                self.metric_lr_reducer.load_state_dict(reducer_state)
                if self.self_lr_scheduler and hasattr(self.lr_scheduler, 'set_external_lr_scale'):
                    self.lr_scheduler.set_external_lr_scale(self.metric_lr_reducer.lr_scale)
            else: 
                self._pending_metric_lr_reducer_state = reducer_state

    def _build_metric_lr_reducer(self):     
        yaml_cfg = getattr(self.cfg, 'yaml_cfg', {})
        reducer_cfg = yaml_cfg.get('metric_lr_reducer', {}) if isinstance(yaml_cfg, dict) else {}   
        if not reducer_cfg or not reducer_cfg.get('enabled', False):    
            if self._pending_metric_lr_reducer_state is not None:     
                logger.warning(
                    'Checkpoint contains metric_lr_reducer_state but current config disables metric_lr_reducer; '
                    'clearing pending reducer state.'  
                )
                self._pending_metric_lr_reducer_state = None  
            self.metric_lr_reducer = None   
            return

        self.metric_lr_reducer = MetricLrReducer( 
            threshold=reducer_cfg.get('threshold', 0.001),
            patience=reducer_cfg.get('patience', 1),    
            factor=reducer_cfg.get('factor', 0.5),
            min_scale=reducer_cfg.get('min_scale', 0.0),
            cooldown=reducer_cfg.get('cooldown', 0), 
            start_epoch=reducer_cfg.get('start_epoch', 0),    
            enabled=reducer_cfg.get('enabled', True),   
        )
  
        if self._pending_metric_lr_reducer_state is not None:  
            self.metric_lr_reducer.load_state_dict(self._pending_metric_lr_reducer_state)
            self._pending_metric_lr_reducer_state = None     
    
        if self.self_lr_scheduler:    
            if hasattr(self.lr_scheduler, 'set_external_lr_scale'):
                self.lr_scheduler.set_external_lr_scale(self.metric_lr_reducer.lr_scale)
            else:
                logger.warning(   
                    'Self LR scheduler is enabled but scheduler does not support set_external_lr_scale; ' 
                    'metric_lr_reducer scale cannot be restored onto scheduler.'     
                )

    def _sync_metric_lr_reducer_with_best_stat(self):
        if self.metric_lr_reducer is None:  
            return
        if not isinstance(getattr(self, 'best_stat', None), dict):
            return
        best_metric = self.best_stat.get('avg_metric', None)  
        if best_metric is None:
            return  
        if self.metric_lr_reducer.best_metric is None or best_metric > self.metric_lr_reducer.best_metric:
            self.metric_lr_reducer.best_metric = best_metric
  
    @staticmethod    
    def _get_avg_ap_metric(test_stats):
        if not test_stats: 
            return None  
        metrics = [test_stats[k][0] for k in test_stats]
        return sum(metrics) / len(metrics) if metrics else None    

    def _apply_metric_lr_reducer(self, test_stats, epoch):  
        if self.metric_lr_reducer is None:     
            return None     

        metric = self._get_avg_ap_metric(test_stats)    
        if metric is None:     
            return None     

        self._sync_metric_lr_reducer_with_best_stat()     
        lr_group_updates = []
        result = self.metric_lr_reducer.step(metric, epoch)
        if self.self_lr_scheduler:    
            if hasattr(self.lr_scheduler, 'set_external_lr_scale'):     
                self.lr_scheduler.set_external_lr_scale(self.metric_lr_reducer.lr_scale)    
            else:   
                logger.warning(  
                    'Self LR scheduler is enabled but scheduler does not support set_external_lr_scale; '    
                    'metric_lr_reducer cannot update scheduler scale.'     
                )
        elif result.get('triggered', False):    
            old_lr_scale = result.get('old_lr_scale', self.metric_lr_reducer.lr_scale)
            if old_lr_scale > 0:
                scale_ratio = result['lr_scale'] / old_lr_scale
                for group_idx, group in enumerate(self.optimizer.param_groups):
                    old_lr = group['lr']    
                    group['lr'] *= scale_ratio
                    lr_group_updates.append((group_idx, old_lr, group['lr']))
                if hasattr(self.lr_scheduler, 'base_lrs'):   
                    self.lr_scheduler.base_lrs = [base_lr * scale_ratio for base_lr in self.lr_scheduler.base_lrs]
                if hasattr(self.lr_scheduler, '_last_lr') and self.lr_scheduler._last_lr is not None: 
                    self.lr_scheduler._last_lr = [last_lr * scale_ratio for last_lr in self.lr_scheduler._last_lr] 
   
        metric_str = f"{metric:.4f}" if metric is not None else "None"
        best_metric = result.get('best_metric', None)   
        best_metric_str = f"{best_metric:.4f}" if best_metric is not None else "None"    
        improvement = result.get('improvement', None)    
        improvement_str = f"{improvement:.4f}" if improvement is not None else "None"    
        status_suffix = ""
        if epoch < self.metric_lr_reducer.start_epoch:
            status_suffix = (
                f" status=inactive waiting_for_start_epoch={self.metric_lr_reducer.start_epoch}"
            ) 
        logger.info(
            "[MetricLrReducer] "
            f"epoch={epoch} metric={metric_str} best={best_metric_str} "  
            f"improvement={improvement_str} bad_epochs={result['bad_epochs']} "    
            f"lr_scale={result['lr_scale']:.4f} triggered={result['triggered']}"   
            f"{status_suffix}" 
        )
        if result.get('triggered', False):  
            old_lr_scale = result.get('old_lr_scale', self.metric_lr_reducer.lr_scale)  
            logger.info(    
                YELLOW
                + (     
                    f"[MetricLrReducer][Triggered] epoch={epoch} "
                    f"lr_scale {old_lr_scale:.4f} -> {result['lr_scale']:.4f}"    
                )
                + RESET   
            )    
            for group_idx, old_lr, new_lr in lr_group_updates:  
                logger.info(
                    YELLOW     
                    + (  
                        f"[MetricLrReducer][ParamGroup] epoch={epoch} "     
                        f"param_group[{group_idx}] lr {old_lr:.6f} -> {new_lr:.6f}"   
                    )
                    + RESET
                )
        return result     

    def _build_baseline_log_monitor(self, total_epochs):  
        yaml_cfg = getattr(self.cfg, 'yaml_cfg', {})  
        monitor_cfg = yaml_cfg.get('baseline_log_monitor', {}) if isinstance(yaml_cfg, dict) else {} 
        self.baseline_log_monitor = BaselineLogMonitor.from_config(monitor_cfg, total_epochs)
        if self.baseline_log_monitor is not None:    
            logger.info(
                "[BaselineLogMonitor] "
                f"enabled metric={self.baseline_log_monitor.metric} "
                f"start_epoch={self.baseline_log_monitor.start_epoch} "  
                f"min_ap_gap={self.baseline_log_monitor.min_ap_gap:.4f} "   
                f"patience={self.baseline_log_monitor.patience}"
            ) 

    @staticmethod  
    def _format_baseline_value(value):
        return "None" if value is None else f"{value:.4f}"

    def _baseline_log_color(self, decision): 
        if decision.diff is None:    
            return RESET    
        if decision.diff >= 0:    
            return GREEN  
        if decision.diff < -self.baseline_log_monitor.min_ap_gap:
            return RED
        return YELLOW   
 
    def _check_baseline_log_monitor(self, log_stats, epoch): 
        if self.baseline_log_monitor is None:    
            return None
 
        decision = self.baseline_log_monitor.check(log_stats, epoch)
        color = self._baseline_log_color(decision)    
        status_suffix = ""
        if not decision.active:     
            status_suffix = f" status=inactive {decision.reason}"
        elif decision.reason and not decision.should_stop:  
            status_suffix = f" status=skip reason={decision.reason}"
     
        logger.info(
            color
            + (
                "[BaselineLogMonitor] "
                f"epoch={epoch} metric={decision.metric} "
                f"current={self._format_baseline_value(decision.current)} "
                f"baseline={self._format_baseline_value(decision.baseline)} "  
                f"diff={self._format_baseline_value(decision.diff)} "  
                f"bad_epochs={decision.bad_epochs}/{decision.patience}"    
                f"{status_suffix}"   
            )
            + RESET 
        )    

        if decision.should_stop:
            logger.warning(
                YELLOW   
                + (
                    f"[BaselineLogMonitor][EarlyStop] epoch={epoch} "
                    f"{decision.reason}"
                )  
                + RESET   
            )    
        return decision    
 
    def _build_checkpoint_paths(self, epoch, checkpoint_freq):
        if not self.output_dir:
            return []
   
        checkpoint_paths = [self.output_dir / 'last.pth'] 
        stop_epoch = getattr(self.train_dataloader.collate_fn, 'stop_epoch', None)  
        if stop_epoch is not None and epoch < stop_epoch and (epoch + 1) % checkpoint_freq == 0:
            checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
        return checkpoint_paths

    def _load_best_stg1_if_available(self):     
        if not self.output_dir:
            logger.warning('Skip loading best_stg1.pth because output_dir is empty.') 
            return False 
 
        stg1_model_path = self.output_dir / 'best_stg1.pth'     
        if not stg1_model_path.exists():   
            logger.warning(f'Skip loading stage1 best model because file does not exist: {stg1_model_path}')
            return False

        self.load_resume_state(str(stg1_model_path))
        return True  

    def _bootstrap_best_stat_from_eval(self, test_stats, epoch):
        if not test_stats:    
            return    
    
        self._ensure_best_stat()   
        if 'avg_metric' in self.best_stat: 
            logger.info(f'Keep resumed best_stat baseline: {self.best_stat}')
            return   

        metrics = {k: test_stats[k][0] for k in test_stats}  
        avg_metric = sum(metrics.values()) / len(metrics)
        self.best_stat['epoch'] = epoch
        self.best_stat['avg_metric'] = avg_metric
        self.best_stat.update(metrics)   

        logger.info(f'Resume from epoch {epoch}:')    
        logger.info(f'  Avg metric: {avg_metric:.4f}')
        for k, v in metrics.items(): 
            logger.info(f'  {k}: {v:.4f}')  
        logger.info(f'Initialized best_stat from resume eval: {self.best_stat}')     

    def fit(self, cfg_str):   
        self.train()   
        args = self.cfg
  
        if dist_utils.is_main_process(): 
            with open(self.output_dir / 'args.json', 'w') as json_file:     
                json_file.write(cfg_str)     

        # 计算模型参数量、FLOPs 等统计信息 
        n_parameters, model_stats = stats(self.cfg)
        print(model_stats)    
        # print("-"*42 + "Start training" + "-"*43)    
        logger.info("Start training")
  
        # 初始化学习率调度器   
        self.self_lr_scheduler = False    
        if args.lrsheduler is not None:
            iter_per_epoch = len(self.train_dataloader)
            # print("     ## Using Self-defined Scheduler-{} ## ".format(args.lrsheduler))    
            logger.info("     ## Using Self-defined Scheduler-{} ## ".format(args.lrsheduler))   
            self.lr_scheduler = FlatCosineLRScheduler(self.optimizer, args.lr_gamma, iter_per_epoch, total_epochs=args.epoches,  
                                                warmup_iter=args.warmup_iter, flat_epochs=args.flat_epoch, no_aug_epochs=args.no_aug_epoch, lr_scyedule_save_path=self.output_dir)
            self.self_lr_scheduler = True    
        self._build_metric_lr_reducer()
        self._build_baseline_log_monitor(args.epoches)   
        # 统计需要训练的参数数量 
        n_parameters = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        # print(f'number of trainable parameters: {n_parameters}')  
        logger.info(f'number of trainable parameters: {n_parameters}')

        self.criterion.set_model(self.model)
        self._ensure_best_stat()   
        # evaluate again before resume training
        if self.last_epoch > 0:    
            module = self.ema.module if self.ema else self.model   
            test_stats, coco_evaluator = evaluate(
                module,  
                self.criterion,
                self.postprocessor, 
                self.val_dataloader,  
                self.evaluator,
                self.device,
                yolo_metrice=self.cfg.yolo_metrice  
            )   
            self._bootstrap_best_stat_from_eval(test_stats, epoch=self.last_epoch)  
            self._sync_metric_lr_reducer_with_best_stat()
 
        start_time = time.time() 
        start_epoch = self.last_epoch + 1   
        for epoch in range(start_epoch, args.epoches):

            self.train_dataloader.set_epoch(epoch)
            self.criterion.set_epoch(epoch)     
            if hasattr(self.train_dataloader.sampler, 'set_epoch'):
                self.train_dataloader.sampler.set_epoch(epoch)  

            if epoch == self.train_dataloader.collate_fn.stop_epoch:
                with ModelSaverFunc(self):
                    self._load_best_stg1_if_available()  
                self.ema.decay = self.train_dataloader.collate_fn.ema_restart_decay   
                # print(f'Refresh EMA at epoch {epoch} with decay {self.ema.decay}')
                logger.info(f'Refresh EMA at epoch {epoch} with decay {self.ema.decay}')   

            # 训练一个 epoch
            train_stats = train_one_epoch(
                self.self_lr_scheduler,
                self.lr_scheduler, 
                self.model, 
                self.criterion,     
                self.train_dataloader,     
                self.optimizer, 
                self.device, 
                epoch, 
                max_norm=args.clip_max_norm, 
                print_freq=args.print_freq,   
                ema=self.ema, 
                scaler=self.scaler,     
                lr_warmup_scheduler=self.lr_warmup_scheduler,  
                writer=self.writer,
                plot_train_batch_freq=args.plot_train_batch_freq,
                output_dir=self.output_dir,     
                epoches=args.epoches, # 总的训练次数
                verbose_type=args.verbose_type    
            )     
    
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            if not self.self_lr_scheduler:  # update by epoch  
                if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                    self.lr_scheduler.step()
 
            self.last_epoch += 1     

            # 训练一个epoch后计算模型指标
            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,  
                self.criterion, 
                self.postprocessor,    
                self.val_dataloader,
                self.evaluator,
                self.device,
                yolo_metrice=self.cfg.yolo_metrice
            )
     
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()   
            
            self._apply_metric_lr_reducer(test_stats, epoch)
            self.best_stat = self.save_best_model(test_stats, self.best_stat, epoch)

            checkpoint_paths = self._build_checkpoint_paths(epoch, args.checkpoint_freq)
            if checkpoint_paths:
                with ModelSaverFunc(self):
                    for checkpoint_path in checkpoint_paths:
                        dist_utils.save_on_master(self.state_dict(), checkpoint_path)
 
            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},   
                **{f'test_{k}': v for k, v in test_stats.items()},
                'epoch': epoch,
                'n_parameters': n_parameters    
            }

            if self.output_dir and dist_utils.is_main_process():   
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")    
  
                # for evaluation logs  
                # if coco_evaluator is not None: 
                #     (self.output_dir / 'eval').mkdir(exist_ok=True)     
                #     if "bbox" in coco_evaluator.coco_eval:  
                #         filenames = ['latest.pth']   
                #         if epoch % 50 == 0:    
                #             filenames.append(f'{epoch:03}.pth')     
                #         for name in filenames:   
                #             torch.save(coco_evaluator.coco_eval["bbox"].eval,   
                #                     self.output_dir / "eval" / name)    
     
            baseline_decision = self._check_baseline_log_monitor(log_stats, epoch)
            if baseline_decision is not None and baseline_decision.should_stop:
                break

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info('Training time {}'.format(total_time_str)) 


    def val(self, ):    
        self.eval()     

        module = self.ema.module if self.ema else self.model
        module.deploy()
        _, model_info = stats(self.cfg, module=module)
        logger.info(GREEN + f"Model Info(fused) {model_info}" + RESET)
        get_weight_size(module)  
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                self.val_dataloader, self.evaluator, self.device, True, self.output_dir, self.cfg.yolo_metrice)

        if self.output_dir:
            dist_utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")  

        return   
    
    def _load_onnx_model(self):   
        """加载ONNX模型"""   
        import onnxruntime as ort   
     
        logger.info(f"Loading ONNX Model: {self.cfg.path}") 
        model = ort.InferenceSession(self.cfg.path)     
        logger.info(f"Using device: {ort.get_device()}")
    
        return {'onnx': model} 

    def _load_engine_model(self, mode):
        """加载TensorRT Engine模型"""
        if mode == 'det': 
            from tools.inference.detect.trt_inf import TRTInference
        elif mode == 'mask':    
            from tools.inference.segment.trt_inf import TRTInference
        else: 
            raise ValueError(f"不支持的模式: {mode}")   
     
        logger.info(f"Loading Engine Model: {self.cfg.path}")
        model = TRTInference(self.cfg.path, device=self.device)  
        logger.info(f"Using device: {self.device}")
        
        return {'engine': model} 
  
    def _load_pth_model(self):  
        """加载PyTorch模型"""
        logger.info(f"Loading PyTorch Model: {self.cfg.path}") 
        
        state = torch.load(self.cfg.path, map_location='cpu', weights_only=False)   
        module = state['prune_model'].to(self.device)
        module.deploy()
        
        # 打印模型信息
        _, model_info = stats(self.cfg, module=module)
        logger.info(GREEN + f"Model Info(fused) {model_info}" + RESET)  
        get_weight_size(module)
   
        return module

    def val_pt_onnx_engine(self, mode):
 
        if self.cfg.path.endswith('onnx') or self.cfg.path.endswith('engine'):  
            self.cfg.yaml_cfg['val_dataloader']['total_batch_size'] = 1     
            self.cfg.yaml_cfg['eval_mask_ratio'] = 1
            logger.warning(RED + f"仅支持batch_size=1进行验证" + RESET)     

        self.eval()
        # 根据模型格式加载不同的模型
        if self.cfg.path.endswith('onnx'):
            model = self._load_onnx_model()
        elif self.cfg.path.endswith('engine'):
            model = self._load_engine_model(mode)   
        elif self.cfg.path.endswith('pth'):
            model = self._load_pth_model()
        else:
            raise ValueError(f"不支持的模型格式: {self.cfg.path}")  
    
        # 执行评估  
        test_stats, coco_evaluator = evaluate(
            model if isinstance(model, torch.nn.Module) else None,
            self.criterion, 
            self.postprocessor, 
            self.val_dataloader,     
            self.evaluator,     
            self.device,   
            True,
            self.output_dir,     
            self.cfg.yolo_metrice,    
            model if not isinstance(model, torch.nn.Module) else None   
        )     
    
        if self.output_dir:
            dist_utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")    
   
        return    
 
    def save_best_model(self, test_stats, best_stat, epoch):    
        if test_stats:     
            # 计算所有评判标准的平均AP（用于判断最佳模型）
            all_metrics = [] 
            metric_names = []
            for k in test_stats:  
                if self.writer and dist_utils.is_main_process():  
                    if getattr(self.evaluator, 'using_vtiny_metrice', False):   
                        coco_metric_names = coco_name_vtiny_list  
                    elif getattr(self.evaluator, 'using_tiny_metrice', False): 
                        coco_metric_names = coco_name_tiny_list
                    else: 
                        coco_metric_names = coco_name_list
                    for i, v in enumerate(test_stats[k]):     
                        self.writer.add_scalar(f'Test/{k}_{coco_metric_names[i]}', v, epoch)  
                
                all_metrics.append(test_stats[k][0]) # 0代表选择ap50-95 1代表选择ap50
                metric_names.append(k)
            
            # 计算平均指标 
            avg_metric = sum(all_metrics) / len(all_metrics) if all_metrics else 0 
    
            # 初始化best_stat  
            if 'avg_metric' not in best_stat:
                best_stat['avg_metric'] = 0     
                best_stat['epoch'] = -1 
                # 为每个指标初始化  
                for k in metric_names:    
                    best_stat[k] = 0
            
            # 保存旧的最佳值（用于日志输出）
            best_stat_temp = best_stat.copy()

            # 判断是否是新的最佳模型    
            is_best = avg_metric > best_stat['avg_metric']
 
            if is_best:
                best_stat['epoch'] = epoch    
                best_stat['avg_metric'] = avg_metric
                # 更新每个指标的最佳值   
                for k, metric_val in zip(metric_names, all_metrics):
                    best_stat[k] = metric_val
    
            # 日志输出   
            logger.info(f'Current metrics: {dict(zip(metric_names, all_metrics))}')
            logger.info(f'Current avg: {avg_metric:.4f}, Best avg: {best_stat["avg_metric"]:.4f} (epoch {best_stat["epoch"]})') 

            # 保存最佳模型
            if is_best and self.output_dir:
                logger.info(RED + f"🎉 New Best Model!" + RESET)   
                logger.info(RED + f"  Epoch: {best_stat_temp['epoch']} -> {best_stat['epoch']}" + RESET)
                logger.info(RED + f"  Avg AP: {best_stat_temp.get('avg_metric', 0):.4f} -> {best_stat['avg_metric']:.4f}" + RESET)
                
                # 打印每个指标的变化 
                for k in metric_names:  
                    old_val = best_stat_temp.get(k, 0)
                    new_val = best_stat[k]
                    logger.info(RED + f"  {k}: {old_val:.4f} -> {new_val:.4f}" + RESET)  
    
                with ModelSaverFunc(self):
                    # 根据训练阶段保存不同的模型   
                    if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                        save_path = self.output_dir / f'best_stg2.pth'
                        dist_utils.save_on_master(self.state_dict(), save_path)   
                        logger.info(RED + f"💾 Saved best_stg2.pth" + RESET) 
                    else:    
                        save_path = self.output_dir / f'best_stg1.pth'  
                        dist_utils.save_on_master(self.state_dict(), save_path) 
                        logger.info(RED + f"💾 Saved best_stg1.pth" + RESET)
  
            # Stage 2 开始时的特殊处理  
            elif epoch >= self.train_dataloader.collate_fn.stop_epoch and epoch == self.train_dataloader.collate_fn.stop_epoch:
                self.ema.decay -= 0.0001  # 衰减因子变小意味着当前模型参数在EMA更新中的占比更大     
                with ModelSaverFunc(self):
                    loaded = self._load_best_stg1_if_available() 
                if loaded:     
                    logger.info(f'🔄 Refresh EMA at epoch {epoch} with decay {self.ema.decay}')
        
        return best_stat     
