import os
import sys
from pathlib import Path

import numpy as np


def write_line(log, message):
    sys.stdout.flush()
    log.write(message + '\n')
    log.flush()


class Report:
    def __init__(self, save_dir, type):
        log_dir = Path(save_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        filename = log_dir / f'{type}_log.txt'
        mode = 'a' if filename.exists() else 'w'
        self.logFile = open(filename, mode)

    def write(self, message):
        print(message)
        write_line(self.logFile, message)

    def __del__(self):
        if hasattr(self, 'logFile') and not self.logFile.closed:
            self.logFile.close()


class Train_Report:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_loss = []
        self.diff_loss = []
        self.triplet_loss = []
        self.fm_loss = []
        self.center_loss = []
        self.num_examples = 0

    def _update_common(self, batch_size, total_loss, triplet_loss, center_loss):
        self.num_examples += batch_size
        self.total_loss.append(total_loss * batch_size)
        self.triplet_loss.append(triplet_loss * batch_size)
        self.center_loss.append(center_loss * batch_size)

    def update(self, batch_size, total_loss, fm_loss, triplet_loss, center_loss):
        self._update_common(batch_size, total_loss, triplet_loss, center_loss)
        self.fm_loss.append(fm_loss * batch_size)
        self.diff_loss.append(fm_loss * batch_size)

    def update_center(self, batch_size, total_loss, triplet_loss, center_loss):
        self._update_common(batch_size, total_loss, triplet_loss, center_loss)

    def compute_mean(self):
        self.total_loss = np.sum(self.total_loss) / self.num_examples
        self.fm_loss = np.sum(self.fm_loss) / self.num_examples
        # self.diff_loss = np.sum(self.diff_loss) / self.num_examples
        self.triplet_loss = np.sum(self.triplet_loss) / self.num_examples

        self.center_loss = np.sum(self.center_loss) / self.num_examples

    def result_str(self, lr, period_time):
        self.compute_mean()
        str_out = f'Total Loss: {self.total_loss:.6f}\tLearning rate: {lr:.7f}\tTime: {period_time:.4f}\t'
        str_out += f'Diffusion Loss: {self.diff_loss:.6f}\tTriplet Loss: {self.triplet_loss:.6f}'
        return str_out

    def result_str_fm(self, lr, period_time):
        self.compute_mean()
        str_out = f'Total Loss: {self.total_loss:.6f}\tLearning rate: {lr:.7f}\tTime: {period_time:.4f}\t'
        str_out += f'FlowMatching Loss: {self.fm_loss:.6f}\tTriplet Loss: {self.triplet_loss:.6f}'
        return str_out

    def result_str_center_only(self, lr, period_time):
        self.compute_mean()
        str_out = f'Total Loss: {self.total_loss:.6f}\tLearning rate: {lr:.7f}\tTime: {period_time:.4f}\t'
        str_out += f'Triplet Loss: {self.triplet_loss:.6f}\tCenter Loss: {self.center_loss:.6f}'
        return str_out


    def result_str_cl(self, lr, period_time):
        self.compute_mean()
        str_out = f'Total Loss: {self.total_loss:.6f}\tLearning rate: {lr:.7f}\tTime: {period_time:.4f}\t'
        str_out += f'FlowMatching Loss: {self.fm_loss:.6f}\tTriplet Loss: {self.triplet_loss:.6f}\tCenter Loss: {self.center_loss:.6f}'
        return str_out
