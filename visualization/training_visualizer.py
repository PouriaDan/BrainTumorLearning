import matplotlib.pyplot as plt
from IPython.display import display

class TrainingPlotter:
    def __init__(self, metric_name):
        self.metric_name = metric_name
        self.train_metrics = []
        self.val_metrics = []
        self.train_losses = []
        self.val_losses = []

        self.fig, (self.mtr_ax, self.loss_ax) = plt.subplots(ncols=2)
        self.fig.set_size_inches(9, 4)

        self.mtr_train_line, self.mtr_val_line = self._init_ax(
            self.mtr_ax, xlabel="Epoch", ylabel=self.metric_name, ylim=(0, 1.0)
        )
        self.loss_train_line, self.loss_val_line = self._init_ax(
            self.loss_ax, xlabel="Epoch", ylabel="Loss", ylim=(0,3.0)
        )
        set_axis_spines(self.mtr_ax)
        set_axis_spines(self.loss_ax)
        self.fig.tight_layout()

        self.hfig = display(self.fig, display_id=True)
        self.loss_initialized = False 

    def _init_ax(self, ax, **kwargs):
        train_line, = ax.plot([], [], label="Train", marker='o', color='blue', linewidth=1, markersize=4)
        val_line, = ax.plot([], [], label="Validation", marker='o', color='red', linewidth=1, markersize=4)
        ax.legend()
        ax.yaxis.grid(True)
        ax.set_xticks([0])
        ax.set(**kwargs)
        return train_line, val_line

    def _update_ax(self, ax, train_line, val_line, train_data, val_data, epoch):
        epochs = list(range(1, epoch + 2))
        train_line.set_data(epochs, train_data)
        val_line.set_data(epochs, val_data)

        ax.set_xlim(0, epoch + 2)
        ax.set_xticks(range(0, epoch + 2, max(1, (epoch + 2) // 10)))

    def _update_loss_ylim(self):
        all_losses = self.train_losses + self.val_losses
        max_loss = max(all_losses)

        if not self.loss_initialized or max_loss > self.loss_ax.get_ylim()[1]:
            self.loss_ax.set_ylim(0, max_loss * 1.1)
            self.loss_initialized = True

    def update(self, epoch, train_mtr, val_mtr, train_loss, val_loss):
        self.train_metrics.append(train_mtr)
        self.val_metrics.append(val_mtr)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

        self._update_ax(self.mtr_ax, self.mtr_train_line, self.mtr_val_line,
                        self.train_metrics, self.val_metrics, epoch)
        self._update_ax(self.loss_ax, self.loss_train_line, self.loss_val_line,
                        self.train_losses, self.val_losses, epoch)

        self._update_loss_ylim()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self.hfig.update(self.fig)

    def save_plot(self, path):
        self.fig.savefig(path)

    def close(self):
        plt.close(self.fig)

def set_axis_spines(ax, left=True, bottom=True, right=False, top=False):
    ax.spines['left'].set_visible(left)
    ax.spines['bottom'].set_visible(bottom)
    ax.spines['right'].set_visible(right)
    ax.spines['top'].set_visible(top)
