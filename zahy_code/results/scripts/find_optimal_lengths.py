from sys import argv
from csv import reader
from os import listdir


"""
Input:

subject, instance, optimal_length, human_length, complete, start_time, end_time, rt,nodes_expanded, skipped, trial_number
A39GADIK8RLMVC:36AHBNMV1SKT9O0GSS6XX0QHZV3YDN,prb9718,999,32,False,1.51630404766e+12,1.51630407583e+12,28169.0,999,False,0
A39GADIK8RLMVC:36AHBNMV1SKT9O0GSS6XX0QHZV3YDN,prb29585,999,15,False,1.51630420294e+12,1.51630421924e+12,16295.0,999,False,1

Output:
subject, instance, optimal_length, human_length, complete, start_time, end_time, rt,nodes_expanded, skipped, trial_number
A39GADIK8RLMVC:36AHBNMV1SKT9O0GSS6XX0QHZV3YDN,prb9718,<REAL_ONE>,32,False,1.51630404766e+12,1.51630407583e+12,28169.0,999,False,0
A39GADIK8RLMVC:36AHBNMV1SKT9O0GSS6XX0QHZV3YDN,prb29585,<REAL_ONE>,15,False,1.51630420294e+12,1.51630421924e+12,16295.0,999,False,1

json_file_name: 
    1509552893_29_prb32120_9_.json
"""

path_file = argv[1]
json_dir = '../../psiturk-rushhour/static/json'
jsons=listdir(json_dir)
jsons=[j for j in jsons if j.endswith('.json')]
jsons=dict([(j.split('_')[2],j.split('_')[3]) for j in jsons])
jsons[' instance']='optimal_length'

with open(path_file,'r') as f:
    for p in reader(f):
        file_name=p[1]
        p[2]=jsons[file_name]
        print ','.join(p)


