# this file imports custom routes into the experiment server

from flask import Blueprint, render_template, request, jsonify, Response, abort, current_app
from time import time
from jinja2 import TemplateNotFound
from functools import wraps
from sqlalchemy import or_
from random import sample
from collections import namedtuple
import re

from psiturk.psiturk_config import PsiturkConfig
from psiturk.experiment_errors import ExperimentError
from psiturk.user_utils import PsiTurkAuthorization, nocache

# # Database setup
from psiturk.db import db_session, init_db
from psiturk.models import Participant
from json import dumps, loads

# load the configuration options
config = PsiturkConfig()
config.load_config()
myauth = PsiTurkAuthorization(config)  # if you want to add a password protect route use this

# explore the Blueprint
custom_code = Blueprint('custom_code', __name__, template_folder='templates', static_folder='static')

###########################################################
#  Stuff needed for the bonus calculation
###########################################################

"""
For now just paste the line from task.js
"""

puzzle_files=[ 'static/json/1509606178_29_prb54081_11_.json','static/json/1513867973_45_prb29232_7_.json', 'static/json/1509722874_99_prb78361_16_.json', 'static/json/1513820498_74_prb1707_7_.json', 'static/json/1509653121_81_prb65535_14_.json', 'static/json/1509546793_06_prb29027_16_.json', 'static/json/1509419634_45_prb343_14_.json', 'static/json/1509473159_82_prb14898_11_.json', 'static/json/1509567550_91_prb38725_14_.json', 'static/json/1509587761_38_prb48146_16_.json', 'static/json/1509682764_39_prb72800_14_.json', 'static/json/1509424444_59_prb3217_11_.json', 'static/json/1513869514_34_prb32795_7_.json', 'static/json/1513854869_31_prb20059_7_.json', 'static/json/1509502982_46_prb25604_16_.json', 'static/json/1509469458_44_prb12715_11_.json', 'static/json/1513866786_91_prb26567_7_.json', 'static/json/1509493367_95_prb23404_14_.json', 'static/json/1509491261_94_prb22436_11_.json', 'static/json/1509637056_7_prb58853_16_.json', 'static/json/1509566883_54_prb38526_11_.json', 'static/json/1509420986_61_prb1267_16_.json', 'static/json/1509547223_69_prb29414_11_.json', 'static/json/1509609106_11_prb54506_16_.json', 'static/json/1509424438_88_prb3203_14_.json', 'static/json/1509411897_18_prb717_11_.json', 'static/json/1513826908_9_prb12604_7_.json', 'static/json/1509503581_83_prb25861_16_.json', 'static/json/1509495640_06_prb24227_16_.json', 'static/json/1509673998_59_prb68514_14_.json', 'static/json/1509556026_72_prb34092_11_.json', 'static/json/1509584113_26_prb45893_16_.json', 'static/json/1509555087_12_prb33509_11_.json', 'static/json/1509585227_74_prb46580_16_.json', 'static/json/1509552512_39_prb31907_11_.json', 'static/json/1513825402_91_prb10206_7_.json', 'static/json/1513827210_11_prb13171_7_.json', 'static/json/1509575509_8_prb42959_11_.json', 'static/json/1513865343_0_prb23259_7_.json', 'static/json/1509547485_18_prb29585_14_.json', 'static/json/1509585277_63_prb46639_16_.json', 'static/json/1513827665_8_prb14047_7_.json', 'static/json/1509584718_51_prb46224_11_.json', 'static/json/1509629349_75_prb57223_16_.json', 'static/json/1513869482_88_prb32695_7_.json', 'static/json/1509643572_34_prb62222_11_.json', 'static/json/1509674597_71_prb68910_11_.json', 'static/json/1513826378_33_prb11647_7_.json', 'static/json/1513867497_06_prb28111_7_.json', 'static/json/1509586671_28_prb47495_14_.json', 'static/json/1509463497_52_prb9718_11_.json', 'static/json/1509556959_89_prb34551_14_.json', 'static/json/1509474295_6_prb15595_16_.json', 'static/json/1509613502_21_prb55384_14_.json', 'static/json/1513821021_88_prb2834_7_.json', 'static/json/1509488403_92_prb20888_14_.json', 'static/json/1509500314_43_prb24406_16_.json', 'static/json/1513824636_72_prb8786_7_.json', 'static/json/1513827972_64_prb14651_7_.json', 'static/json/1509464472_82_prb10166_16_.json', 'static/json/1509724768_91_prb79230_11_.json', 'static/json/1509481067_13_prb19279_14_.json', 'static/json/1509643264_96_prb62015_11_.json', 'static/json/1509457858_6_prb6671_14_.json', 'static/json/1513857327_3_prb21272_7_.json', 'static/json/1513828263_06_prb15290_7_.json', 'static/json/1509577648_29_prb44171_16_.json', 'static/json/1509472355_7_prb14485_14_.json', 'static/json/1509547500_78_prb29600_14_.json', 'static/json/1509554623_5_prb33117_14_.json']

puzzle_files=[ j.split('_')[-3] for j in puzzle_files]
num_of_bonus_puzzles=3
total_bonus=9.0
###########################################################
#  serving warm, fresh, & sweet custom, user-provided routes
#  add them here
###########################################################

#----------------------------------------------
# example custom route
#----------------------------------------------
@custom_code.route('/my_custom_view')
def my_custom_view():
	current_app.logger.info("Reached /my_custom_view")  # Print message to server.log for debugging 
	try:
		return render_template('custom.html')
	except TemplateNotFound:
		abort(404)

#----------------------------------------------
# example using HTTP authentication
#----------------------------------------------
@custom_code.route('/my_password_protected_route')
@myauth.requires_auth
def my_password_protected_route():
	try:
		return render_template('custom.html')
	except TemplateNotFound:
		abort(404)

#----------------------------------------------
# example accessing data
#----------------------------------------------
@custom_code.route('/view_data')
@myauth.requires_auth
def list_my_data():
        users = Participant.query.all()
	try:
		return render_template('list.html', participants=users)
	except TemplateNotFound:
		abort(404)

def add_bonus_data(uniqueId,user,won_puzzles,bonus_puzzles):
    user_data = loads(user.datastring)
    max_trial= int(user_data['data'][-1]['current_trial'])
    for puzzle in bonus_puzzles:
        max_trial+=1
        rec = {"uniqueid":uniqueId,"current_trial":max_trial,"dateTime":time(),"trialdata":'t:['+str(time())+'] event:['+('BONUS_FAIL','BONUS_SUCCESS')[puzzle in won_puzzles]+'] piece:[NA] move#:[NA] move:[NA] instance:['+puzzle+']'}
        user_data['data'].append(rec)
        user.datastring = dumps(user_data)



@custom_code.route('/compute_bonus', methods=['GET'])
def compute_bonus():
    played_puzzles=0;
    if not request.args.has_key('uniqueId'):
        raise ExperimentError('improper_inputs')
    uniqueId = request.args['uniqueId']
    try:
        user = Participant.query.\
               filter(Participant.uniqueid == uniqueId).\
               one()
        user_data = loads(user.datastring) # load datastring from JSON
        won_puzzles=set()
        won_puzzles_numbers=[]
        for record in user_data['data']: # for line in data file
            trial = record['trialdata']
            if isinstance(trial, basestring):
                if 'win' in trial:
                    won_puzzles.add([s.replace('[','').replace(']','') for s in re.findall('\[.*?\]',trial)][5])
        bonus_puzzles=sample(puzzle_files,num_of_bonus_puzzles)
        user.bonus = bonus_by_completion(won_puzzles,bonus_puzzles)
        add_bonus_data(uniqueId,user,won_puzzles,bonus_puzzles)
        db_session.add(user)
        db_session.commit()
        resp = {"bonusComputed": "success"}
        return jsonify(**resp)
    except TemplateNotFound:
        abort(404)
    except Exception as e:
        print 'compute_bonus ERROR:'
        current_app.logger.error(e)  # Print message to server.log for debugging 
        print e



@custom_code.route('/show_bonus', methods=['GET'])
def show_bonus():
    if not request.args.has_key('uniqueId'):
        raise ExperimentError('improper_inputs')
    uniqueId = request.args['uniqueId']
    try:
        user = Participant.query.\
               filter(Participant.uniqueid == uniqueId).\
               one()
        user_data = loads(user.datastring) # load datastring from JSON
        won_puzzles=set()
        succs=[]
        for record in user_data['data']: # for line in data file
            trial = record['trialdata']
            if isinstance(trial, basestring):
                if 'BONUS' in trial:
                    puzzle=[s.replace('[','').replace(']','') for s in re.findall('\[.*?\]',trial)][5]
                    if 'BONUS_SUCCESS' in trial:
                        is_bonus=True
                        amount=total_bonus/num_of_bonus_puzzles
                    elif 'BONUS_FAIL' in trial:
                        is_bonus=False
                        amount=0
                    succs.append((puzzle,is_bonus,amount))
        return render_template('show_bonus.html', success=succs, bonus=user.bonus,time=time())
    except TemplateNotFound:
        abort(404)
    except Exception as e:
        current_app.logger.error(e)  # Print message to server.log for debugging 
        print e


def bonus_by_completion(won_puzzles,bonus_puzzles):
    per_puzzle=total_bonus/num_of_bonus_puzzles
    return len(set(won_puzzles) & set(bonus_puzzles))*per_puzzle

