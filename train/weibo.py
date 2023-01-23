import sys
sys.path.append("..")
from modeling.flat_main import *
from datetime import datetime


grid_sep='='*119+'\n'
line_sep='-'*30+'\n'

stage1_param = {
    'batch': 40,
    'lr':   0.0008,
    'dim': 20,
    'head':10,
    'wd':0.05,
    'warmup':0.1,
    'temp':0.08
}
stage2_param = {
    'batch_2':10,
    'lr_2':0.0004,
    'warmup_2':0.05,
    'wd_2':  0.05,
}





##################### Manual set params #########################
data='weibo'
device='0'
fixed_seed=1080956
#################################################################

is_ctr=False


output_dir = '../runs/' + data + str(datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
writer_file = output_dir  + '/'+str(datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))




writer=open(writer_file,'a+')


args={'data':data,'device':device,}
writer.write(str(args)+'\n\n')
# writer.write('='*53+' Search Grid '+'='*53+'\n')
# writer.write(str(stage1_param)+'\n'+grid_sep)
writer.flush()



# stage1
writer.write('\n'+'-'*27+' Stage1 Parameter '+'-'*27+'\n')
writer.write(str(stage1_param)+'\n')
writer.flush()

metric_ctr,state_path_ctr=flat_main(stage1_param['batch'], stage1_param['lr'], stage1_param['dim'], stage1_param['head'], stage1_param['warmup'],dataset=data,device=device,ctr=True,output_dir=output_dir,weight_decay=stage1_param['wd'],temp=stage1_param['temp'],seed=fixed_seed) # ck=checkpoint

writer.write(str(metric_ctr)+'\n')
writer.write(state_path_ctr+'\n')
writer.write('-' * 72 + '\n\n')
writer.flush()





# stage2
second_metric_head, state_path_head = flat_main(stage2_param['batch_2'], stage2_param['lr_2'],stage1_param['dim'], stage1_param['head'], stage2_param['warmup_2'], dataset=data, device=device,
                               ck=state_path_ctr, output_dir=output_dir,weight_decay=stage2_param['wd_2'],only_head=True,seed=fixed_seed)

second_metric, state_path = flat_main(stage2_param['batch_2'], stage2_param['lr_2'],stage1_param['dim'], stage1_param['head'],stage2_param['warmup_2'], dataset=data, device=device,
                               ck=state_path_head, output_dir=output_dir,weight_decay=stage2_param['wd_2'],only_head=False,seed=fixed_seed)

os.remove(state_path_head)

writer.write(str(second_metric) + '\n')
writer.write(state_path+ '\n\n')
writer.flush()

        # writer.write('best inner_grid f1: '+str(inner_best_score) + '\n')
        # writer.write('achived parameters: '+str(inner_best_hyperparams)+ '\n')
        # writer.write('best state: '+str(inner_best_path)+ '\n\n')
        # writer.flush()


writer.write('finised training!\n')
writer.write('test f1 : '+str(second_metric['f'])+'\n')
writer.flush()