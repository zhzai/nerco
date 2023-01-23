import sys
sys.path.append("..")
from modeling.flat_main_evaluate import *
from datetime import datetime


data='ontonotes'
checkpoint='../checkpoints/onto/Flat_eval_bert-base-chinese_e_24_f_ontonotes_0.836158_24_0.810042.bin'

device='0'

output_dir = '../runs/' + data + str(datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
if not os.path.exists(output_dir):  # 判断是否存在文件夹如果不存在则创建文件夹
    os.makedirs(output_dir)
writer_file = output_dir  + '/'+str(datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
writer=open(writer_file,'a+')

args={'data':data,'device':device,'checkpoint':checkpoint}
writer.write(str(args)+'\n\n')
writer.flush()


inner_metric = flat_main(reinit=0,dataset=data, head_dim=20,head=8,device=device,ck=checkpoint,output_dir=output_dir )


for k,v in inner_metric['SpanFPreRecMetric'].items():
    print(k,v)
writer.write('-' * 72 + '\n')
writer.write(str(inner_metric) + '\n')
writer.write('-' * 72 + '\n\n')
writer.flush()


writer.write('finised evaluation!\n')
writer.write('test f1 : '+str(inner_metric['SpanFPreRecMetric']['f'])+'\n')
writer.flush()