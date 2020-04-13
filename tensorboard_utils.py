from tensorboardX import SummaryWriter
from pathlib import Path
import datetime
            
from tensorboard.backend.event_processing import event_accumulator

def tb_mle_batch(tb, total_loss, n_char_total, n_char_correct, batch_idx):
    tb.add_scalars(
        {
        "loss_per_char" : total_loss / n_char_total,
        "accuracy": n_char_correct / n_char_total,
        },
        group="mle_train",
        #sub_group="validate",
        sub_group="batch",
        global_step = batch_idx)

# def tb_mle_epoch(self, tb, loss_per_char, accuracy, epoch):
#     tb.add_scalars(
#         {
#         "loss_per_char" : loss_per_char,
#         "accuracy" : accuracy,
#         },
#         group="train",
#         sub_group="epoch",
#         global_step=epoch
#     )

def tb_policy_batch(tb, batch_rewards, average_value_loss, batch_idx):
    tb.add_scalars(
      {
        "batch_average_rewards" : batch_rewards,
        "epoch_value_loss": average_value_loss, 
      },
    group="policy_train",
    sub_group="batch",
    global_step = batch_idx)

def tb_policy_epoch(self, tb, average_rewards, average_value_loss, epoch):
    tb.add_scalars(
      {
        "epoch_average_reward" : average_rewards,
        "epoch_value_loss": average_value_loss, 
      },
      group="train",
      sub_group="epoch",
      global_step=epoch
    )
  
def tb_mle_policy_batch(self, tb, total_loss, n_char_total, n_char_correct, batch_rewards, epoch, batch_idx, data_len):
    tb.add_scalars(
    {
        "loss_per_char" : total_loss / n_char_total,
        "accuracy": n_char_correct / n_char_total,
        "batch_average_rewards" : batch_rewards,
    },
    group="mle_policy_train",
    sub_group="batch",
    global_step = epoch*data_len+batch_idx)

def tb_mle_policy_epoch(self, tb, loss_per_char, accuracy, average_rewards, epoch):
    tb.add_scalars(
    {
        "loss_per_char" : loss_per_char,
        "accuracy" : accuracy,
        "epoch_average_reward" : average_rewards,
    },
    group="train",
    sub_group="epoch",
    global_step=epoch
    )

   
def tensorboard_event_accumulator(
    file,
    loaded_scalars=0, # load all scalars by default
    loaded_images=4, # load 4 images by default
    loaded_compressed_histograms=500, # load one histogram by default
    loaded_histograms=1, # load one histogram by default
    loaded_audio=4, # loads 4 audio by default
):
    ea = event_accumulator.EventAccumulator(
        file,
        size_guidance={ # see below regarding this argument
            event_accumulator.COMPRESSED_HISTOGRAMS: loaded_compressed_histograms,
            event_accumulator.IMAGES: loaded_images,
            event_accumulator.AUDIO: loaded_audio,
            event_accumulator.SCALARS: loaded_scalars,
            event_accumulator.HISTOGRAMS: loaded_histograms,
        }
    )
    ea.Reload()
    return ea


class Tensorboard:
    def __init__(
        self,
        experiment_id,
        output_dir="./runs",
        unique_name=None,
    ):
        self.experiment_id = experiment_id
        self.output_dir = Path(output_dir)
        if unique_name is None:
            unique_name = datetime.datetime.now().isoformat(timespec="seconds")
        self.path = self.output_dir / f"{experiment_id}_{unique_name}"
        print(f"Writing TensorBoard events locally to {self.path}")
        self.writers = {}

    def _get_writer(self, group: str=""):
        if group not in self.writers:
            print(
                f"Adding group {group} to writers ({self.writers.keys()})"
            )
            self.writers[group] = SummaryWriter(f"{str(self.path)}_{group}")
        return self.writers[group]
    
    def add_scalars(self, metrics: dict, global_step: int, group=None, sub_group=""):
        for key, val in metrics.items():
            cur_name = "/".join([sub_group, key])
            self._get_writer(group).add_scalar(cur_name, val, global_step)
