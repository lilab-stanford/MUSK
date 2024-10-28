""" Checkpoint Saver

Track top-n training checkpoints and maintain recovery checkpoints on specified intervals.

Hacked together by / Copyright 2020 Ross Wightman
"""

import glob
import operator
import os
import logging
import json #12.24加入

import torch

from .model import unwrap_model, get_state_dict


_logger = logging.getLogger(__name__)


class CheckpointSaver:
    def __init__(
            self,
            model,
            optimizer,
            args=None,
            model_ema=None,
            amp_scaler=None,
            checkpoint_prefix='checkpoint',
            recovery_prefix='recovery',
            checkpoint_dir='',
            recovery_dir='',
            decreasing=False,
            max_history=10,
            unwrap_fn=unwrap_model):

        # objects to save state_dicts of
        self.model = model
        self.optimizer = optimizer
        self.args = args
        self.model_ema = model_ema
        self.amp_scaler = amp_scaler

        # state
        self.checkpoint_files = []  # (filename, metric) tuples in order of decreasing betterness
        self.best_epoch = None
        self.best_metric = None
        self.curr_recovery_file = ''
        self.last_recovery_file = ''
        # 12.20增加，记录最好的best epoch对应的test和单独最好的test
        self.best_metric_val_test = None
        self.best_epoch_test = None 
        self.best_metric_test = None
        #2.5增加
        self.best_f1_test = None #记录best时的f1
        self.best_auc_test = None #记录best时的auc
        self.best_f1 = None #记录val时的f1
        self.best_auc = None #记录val时的auc

        # config
        self.checkpoint_dir = checkpoint_dir
        self.recovery_dir = recovery_dir
        self.save_prefix = checkpoint_prefix
        self.recovery_prefix = recovery_prefix
        self.extension = '.pth.tar'
        self.decreasing = decreasing  # a lower metric is better if True
        self.cmp = operator.lt if decreasing else operator.gt  # True if lhs better than rhs
        self.max_history = max_history
        self.unwrap_fn = unwrap_fn
        assert self.max_history >= 1

    # 12.24 增加一个num_epochs参数，如果相同了，就保存最好10轮的epoch的验证集和测试集结果，并取平均\
    # 2.5 增加了f1和auc
    def save_checkpoint(self, num_epochs, output_dir, epoch, metric=None, metric_test=None, metric_f1=None, metric_auc=None):
        assert epoch >= 0
        tmp_save_path = os.path.join(self.checkpoint_dir, 'tmp' + self.extension)
        last_save_path = os.path.join(self.checkpoint_dir, 'last' + self.extension)
        # self._save(tmp_save_path, epoch, metric) # 1.20不保存任何参数信息
        
        # if os.path.exists(last_save_path):
        #     os.unlink(last_save_path)  # required for Windows support.
        # os.rename(tmp_save_path, last_save_path)
        worst_file = self.checkpoint_files[-1] if self.checkpoint_files else None
        if (len(self.checkpoint_files) < self.max_history
                or metric is None or self.cmp(metric_test, worst_file[2])):  #metric, worst_file[1], 12.20 后面记录test的max #1.22metric->metric_test
            if len(self.checkpoint_files) >= self.max_history:
                self._cleanup_checkpoints(1)
            filename = '-'.join([self.save_prefix, str(epoch)]) + self.extension
            save_path = os.path.join(self.checkpoint_dir, filename)
            # os.link(last_save_path, save_path) #12.20修改，不保存pt了
            # 12.20，增加最优10个的test的metric
            # self.checkpoint_files.append((save_path, metric))
            self.checkpoint_files.append((epoch, metric, metric_test,  metric_f1,  metric_auc)) #12.24直接改为epoch
            self.checkpoint_files = sorted(
                self.checkpoint_files, key=lambda x: x[2], #1，12.20原来按照val高低来排，现在按照test的
                reverse=not self.decreasing)  # sort in descending order if a lower metric is not better

            checkpoints_str = "Current checkpoints:\n"
            for c in self.checkpoint_files:
                checkpoints_str += ' {}\n'.format(c)
            _logger.info(checkpoints_str)

             # 12.24增加最好n轮的平均
            # 每次更新都替换掉之前的json文件，这样随时都可以看得到
            metric_best10_dict = {}
            best10_mean = 0
            best10_f1 = 0
            best10_auc = 0
            for i in range(len(self.checkpoint_files)):
                metric_best10_dict['epoch:{}'.format(self.checkpoint_files[i][0])] = {
                    'acc: ': self.checkpoint_files[i][2], 
                    'f1: ': self.checkpoint_files[i][3], 
                    'auc: ': self.checkpoint_files[i][4] 
                } #0是epoch，2是best, 3是f1,4是auc
                best10_mean += self.checkpoint_files[i][2]
                best10_f1 += self.checkpoint_files[i][3]
                best10_auc += self.checkpoint_files[i][4]
            metric_best10_dict['best10_mean'] = best10_mean / len(self.checkpoint_files)
            metric_best10_dict['best10_f1_mean'] = best10_f1 / len(self.checkpoint_files)
            metric_best10_dict['best10_auc_mean'] = best10_auc / len(self.checkpoint_files)
            submit = os.path.join(output_dir, 'best_10.json')
    #         print(submit)
            with open(submit, 'w') as f: #以写入的方式打开文件，存在则覆盖，不存在则创建
                json.dump(metric_best10_dict, f, indent=2)
            
            if metric is not None and (self.best_metric is None or self.cmp(metric, self.best_metric)):
                self.best_epoch = epoch
                self.best_metric = metric
                self.best_metric_val_test = metric_test #验证集最好时的测试集准确率
                self.best_f1 = metric_f1
                self.best_auc = metric_auc
            if metric_test is not None and (self.best_metric_test is None or self.cmp(metric_test, self.best_metric_test)):
                self.best_epoch_test = epoch
                self.best_metric_test = metric_test #测试集最好准确率
                self.best_f1_test = metric_f1
                self.best_auc_test = metric_auc
            
            #12.20 不保存模型
            # if metric is not None and (self.best_metric is None or self.cmp(metric, self.best_metric)):
            #     self.best_epoch = epoch
            #     self.best_metric = metric
            #     best_save_path = os.path.join(self.checkpoint_dir, 'model_best' + self.extension)
            #     if os.path.exists(best_save_path):
            #         os.unlink(best_save_path)
            #     os.link(last_save_path, best_save_path)

        # return (None, None) if self.best_metric is None else (self.best_metric, self.best_epoch)
        #12.20返回多个值
        return (None, None, None, None, None, None, None, None, None) if self.best_metric is None else (self.best_metric, self.best_metric_val_test, self.best_epoch, self.best_metric_test, self.best_epoch_test, self.best_f1, self.best_auc, self.best_f1_test, self.best_auc_test)

    def _save(self, save_path, epoch, metric=None):
        save_state = {
            'epoch': epoch,
            'arch': type(self.model).__name__.lower(),
            'state_dict': get_state_dict(self.model, self.unwrap_fn),
            'optimizer': self.optimizer.state_dict(),
            'version': 2,  # version < 2 increments epoch before save
        }
        if self.args is not None:
            save_state['arch'] = self.args.model
            save_state['args'] = self.args
        if self.amp_scaler is not None:
            save_state[self.amp_scaler.state_dict_key] = self.amp_scaler.state_dict()
        if self.model_ema is not None:
            save_state['state_dict_ema'] = get_state_dict(self.model_ema, self.unwrap_fn)
        if metric is not None:
            save_state['metric'] = metric
        torch.save(save_state, save_path) #1.20 tmp.pth也不需要保存

    def _cleanup_checkpoints(self, trim=0):
        trim = min(len(self.checkpoint_files), trim)
        delete_index = self.max_history - trim
        if delete_index < 0 or len(self.checkpoint_files) <= delete_index:
            return
        to_delete = self.checkpoint_files[delete_index:]
        for d in to_delete:
            try:
                _logger.debug("Cleaning checkpoint: {}".format(d))
                os.remove(d[0])
            except Exception as e:
                _logger.error("Exception '{}' while deleting checkpoint".format(e))
        self.checkpoint_files = self.checkpoint_files[:delete_index]

    def save_recovery(self, epoch, batch_idx=0):
        assert epoch >= 0
        filename = '-'.join([self.recovery_prefix, str(epoch), str(batch_idx)]) + self.extension
        save_path = os.path.join(self.recovery_dir, filename)
        self._save(save_path, epoch)
        if os.path.exists(self.last_recovery_file):
            try:
                _logger.debug("Cleaning recovery: {}".format(self.last_recovery_file))
                os.remove(self.last_recovery_file)
            except Exception as e:
                _logger.error("Exception '{}' while removing {}".format(e, self.last_recovery_file))
        self.last_recovery_file = self.curr_recovery_file
        self.curr_recovery_file = save_path

    def find_recovery(self):
        recovery_path = os.path.join(self.recovery_dir, self.recovery_prefix)
        files = glob.glob(recovery_path + '*' + self.extension)
        files = sorted(files)
        return files[0] if len(files) else ''
