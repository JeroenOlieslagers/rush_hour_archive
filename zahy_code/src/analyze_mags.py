"""
Analysis for MAGS.

"""
import os
from collections import namedtuple
from rushhour import instance_dict, do_move_from_fixed, constuct_mag, draw
from rushhour import find_piece
from rushhour import mag2dot

_tmp_path = os.path.dirname(os.path.abspath(__file__)).split('/')

"""
Make sure to set this to the project name if it was renamed  
"""
PROJECT_NAME = 'rushhour'

PROJECT_ROOT = '/'.join(_tmp_path[:_tmp_path.index(PROJECT_NAME)+1])
PROBLEM_FOLDER = PROJECT_ROOT + '/psiturk-rushhour/static/json'
FIGURE_FOLDER = PROJECT_ROOT + '/results/figures/'
MOVES_FILE = PROJECT_ROOT + '/results/all_stages/moves.csv'
MOVEDATA_FILE = PROJECT_ROOT + '/results/all_stages/movedata.csv'
RAW_FILE = PROJECT_ROOT + '/results/all_stages/trialdata.csv'
MOVE_FILE_WITH_MAG = PROJECT_ROOT + '/results/all_stages/moves_with_mag.csv'
MOVE_FILE_FIELDS = ['subject', 'instance', 'optimal_length',
                    'move_number', 'move', 'pre_actions', 'meta_move',
                    'rt', 'trial_number', 'progress', 'distance_to_goal']

MoveFileRec = namedtuple('MoveFileRec', ' '.join(MOVE_FILE_FIELDS))


def to_rec(string):
    """
    :param string:
    :return:  Rec from String
    """
    return MoveFileRec(*(string.strip().split(',')))


def is_initial_state(rec):
    return int(rec.move_number) == 0


def mag_stats(rec, state, piece, location, show_mag=True):
    """
    Mag Statistics are added here
    :param rec:
    :param state:
    :param piece:
    :param location:
    :return:
    """

    stats = {}
    mag, mag_nodes = constuct_mag(state)
    if show_mag:
        mag2dot(mag, True)

    """
    Nodes in the MAG (including leaves)
    """
    number_of_nodes = len(mag_nodes)
    stats['number_of_nodes'] = number_of_nodes

    """
    Cars that can move now 
    """
    number_of_leaves = number_of_nodes - (len(mag) - 1)# because of Dummy node
    stats['number_of_leaves'] = number_of_leaves





    return stats






def analyze_mags(logging=False):
    """
    Entry point for analysis of mags

    Goes over move data.
        1. Reconstruct state
        2. Build Mag
        3. Report Mag Features
    :return:
    """
    instances = instance_dict()
    state = None
    header = None
    with open(MOVES_FILE, 'r') as input_file,\
            open(MOVE_FILE_WITH_MAG, 'w') as output_file:
        for line in input_file:
            line = line.strip()
            if line.startswith('subject'):
                header = line
                continue
            rec = to_rec(line)
            if is_initial_state(rec):
                state = instances[rec.instance]
            piece, one_d_position = rec.move.split('@')
            if one_d_position == 'undefined':
                continue
            (_, _, size), _ = find_piece(state, piece)
            if logging:
                print('piece:' + str(piece) + ' location:' + str(one_d_position))
            location = ((int(one_d_position) % 6),
                        int(one_d_position) / 6,
                        size)
            mag_statistics = mag_stats(rec, state, piece, location)
            for mag_stat_field in sorted(mag_statistics):
                line += ',' + str(mag_statistics[mag_stat_field])

            if logging: draw(state)
            do_move_from_fixed(state, (piece, location))
            if logging: draw(state)

            if header is not None:
                header += ',' + ','.join(sorted(mag_statistics))
                print(header)
                output_file.write(header + '\r\n')
                header = None
            print(line)
            output_file.write(line + '\r\n')


if __name__ == '__main__':
    analyze_mags()
