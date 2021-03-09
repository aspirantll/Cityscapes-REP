from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
import os
import shutil
import time
import sys
import torch

USE_TENSORBOARD = True
try:
    import tensorboardX

    print('Using tensorboardX')
except:
    USE_TENSORBOARD = False


class SimpleLogger(object):
    def __init__(self, opt):
        self.start_line = True
        self._opt = opt

        args = dict((name, getattr(opt, name)) for name in dir(opt)
                    if not name.startswith('_'))
        file_name = os.path.join(opt.save_dir, 'opt.txt')
        with open(file_name, 'w+') as opt_file:
            opt_file.write('==> torch version: {}\n'.format(torch.__version__))
            opt_file.write('==> cudnn version: {}\n'.format(
                torch.backends.cudnn.version()))
            opt_file.write('==> Cmd:\n')
            opt_file.write(str(sys.argv))
            opt_file.write('\n==> Opt:\n')
            for k, v in sorted(args.items()):
                opt_file.write('  %s: %s\n' % (str(k), str(v)))

        log_dir = opt.save_dir + '/logs'
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        self.log_dir = log_dir
        self.summary_count = 0

    def write(self, txt='', end='\n', level=0):
        txt = "\t"* level + str(txt) + end
        if self.start_line:
            time_str = time.strftime('%Y-%m-%d %H:%M:%S')
            self.write_txt('[{}] {}'.format(time_str, txt))
        else:
            self.write_txt(txt)
        self.start_line = False
        if '\n' in txt:
            self.start_line = True
            self.flush()

    def write_txt(self, txt):
        print(txt, end="")

    def open_summary_writer(self):
        if USE_TENSORBOARD:
            self.writer = tensorboardX.SummaryWriter(log_dir=self.log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        if USE_TENSORBOARD:
            self.writer.add_scalar(tag, value, step)

    def flush(self):
        pass

    def close_summary_writer(self):
        if USE_TENSORBOARD:
            self.writer.close()

    def close(self):
        pass


class Logger(SimpleLogger):
    _logger = None

    def __init__(self, opt):
        """Create a summary writer logging to log_dir."""
        super().__init__(opt)

        self.log_root_path = self.log_dir + '/log.txt'
        self.log = open(self.log_root_path, 'w+')

    def write_txt(self, txt):
        super().write_txt(txt)
        self.log.write(txt)

    def close(self):
        self.log.close()



    @staticmethod
    def init_logger(opt, type="logger"):
        if Logger._logger is None:
            if type == "simple":
                Logger._logger = SimpleLogger(opt)
            else:
                Logger._logger = Logger(opt)

    @staticmethod
    def get_logger():
        return Logger._logger
