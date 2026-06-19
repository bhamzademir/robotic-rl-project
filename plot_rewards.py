import os
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt

log_dir = "ppo_franka_tensorboard/PPO_1/"
event_file = [f for f in os.listdir(log_dir) if "events.out.tfevents" in f][0]
file_path = os.path.join(log_dir, event_file)

ea = event_accumulator.EventAccumulator(file_path)
ea.Reload()

# Mevcut tüm etiketleri (tags) ekrana basalım ki ne var ne yok görelim
print("Available metrics in log file:")
print(ea.Tags()['scalars'])

# Genel eğitim kaybını (loss) çizdirelim (Bu kesinlikle vardır)
if 'train/loss' in ea.Tags()['scalars']:
    loss_data = ea.Scalars('train/loss')
    steps = [x.step for x in loss_data]
    values = [x.value for x in loss_data]

    plt.figure(figsize=(10, 5))
    plt.plot(steps, values, label='Training Loss', color='red', linewidth=2)
    plt.xlabel('Timesteps')
    plt.ylabel('Loss')
    plt.title('PPO Training Loss - Franka Panda')
    plt.grid(True)
    plt.legend()
    
    plt.savefig('training_loss_curve.png')
    print("\n✅ Loss curve successfully saved as 'training_loss_curve.png'!")
    plt.show()
else:
    print("\nNo scalar metrics found yet. The logs might still be buffering in memory.")