import torch as t
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset import LoadDataset
from torch import optim
from Models import Deeplab_v3plus
import cfg
from metrics import averageMeter, runningScore
import time

device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
num_class = cfg.DATASET[1]

BATCH_SIZE = 1
miou_list = [0]

Load_test = LoadDataset([cfg.TEST_ROOT, cfg.TEST_LABEL], cfg.crop_size)
test_data = DataLoader(Load_test, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

net = Deeplab_v3plus.DeepLabv3_plus(n_InputChannels=3, n_classes=cfg.DATASET[1], os=16, _print=False)
net.eval()
net.to(device)
net.load_state_dict(t.load('./Results/weights/FCN_weight/98.pth'))
running_metrics_val = runningScore(10)
time_meter = averageMeter()

for i, sample in enumerate(test_data):
	time_start = time.time()
	data = Variable(sample['img']).to(device)
	label = Variable(sample['label']).to(device)
	out = net(data)
	out = F.log_softmax(out, dim=1)

	pre_label = out.max(dim=1)[1].data.cpu().numpy()
	true_label = label.data.cpu().numpy()
	running_metrics_val.update(true_label, pre_label)
	time_meter.update(time.time() - time_start, n=data.size(0))

metrics = running_metrics_val.get_scores()
for k, v in metrics[0].items():
	print(k, v)
print('inference time per image: ', time_meter.avg)
print('inference fps: ', 1 / time_meter.avg)