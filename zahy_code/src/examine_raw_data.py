from collections import defaultdict

import matplotlib
from analyze import read_psiturk_data
from analyze_mags import RAW_FILE, FIGURE_FOLDER

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

"""
Examine what data to use. 
This is where filters should be implemented
"""

def plot_time_line_data_per_worker(worker_recs):
    print('sorting by time')
    order_by_t = sorted(worker_recs, key=lambda r: float(r.t))

    print('getting times')
    times = [float(r.t) for r in order_by_t]

    print('getting events')
    events = [r.event for r in order_by_t]

    print('getting instances timeline')
    instances = {}
    for r in order_by_t:
        if r.instance not in instances:
            instances[r.instance] = float(r.t)

    fig, axs = plt.subplots(nrows=2, sharex='all')
    times_ints = [int(t) for t in times]
    #plt.xticks(range(min(times_ints), max(times_ints), max(int((max(times_ints) - min(times_ints))/5), 1)))

    print('*Timeline for worker:' + str(worker_recs[0].worker))
    fig.suptitle('Timeline for worker:' + str(worker_recs[0].worker), fontsize=16)
    axs[0].set_title('puzzles')
    axs[0].set_xlabel("time")
    axs[0].set_ylabel("Instance")

    axs[1].set_title('events')
    axs[1].set_xlabel("time")
    axs[1].set_ylabel("event")

    print('plotting instances')

    axs[0].scatter(list(instances.values()), list(instances.keys()))

    print('plotting events')
    axs[1].scatter(times, events)
    #plt.savefig(FIGURE_FOLDER + 'timeline_' + str(worker_recs[0].worker)+'.pdf')



def plot_times(by_worker, worker_id):
    worker_recs = by_worker[worker_id]
    order_by_t = sorted(worker_recs, key=lambda r: float(r.t))
    times = [float(r.t) for r in order_by_t]
    plt.scatter(range(len(order_by_t)), times)
    plt.show()


def filter_weird(recs):
    with open(FIGURE_FOLDER +'Weird_subjects', 'r') as f:
        bad_workers = [w.strip() for w in f.readlines()]
    return [r for r in recs if r.worker not in bad_workers]

def filter_weird_on_files(file_to_filter):
    with open(FIGURE_FOLDER +'Weird_subjects', 'r') as f:
        bad_workers = [w.strip() for w in f.readlines()]

    with open(file_to_filter, 'r') as input:
        with open(file_to_filter+'_filtered', 'wa') as output:
            for l in input:
               if not any(w in l for w in bad_workers):
                   print(l)
                   output.write(l)




if __name__ == '__main__':
    """
    PsiturkRec=namedtuple('PsiturkRec','worker assignment ord t event piece move_nu move instance')
    """
    recs = filter_weird(read_psiturk_data(RAW_FILE))
    by_worker = defaultdict(list)


    print('splitting by worker.')
    for r in recs:
        by_worker[r.worker].append(r)

    for w in by_worker:
        print(str(w) + ' has ' + str(len(by_worker[w])) + ' records')

    plot_times(by_worker, 'A3NMQ3019X6YE0')
    duplicate_worker = by_worker['A3NMQ3019X6YE0']
    by_worker['A3NMQ3019X6YE0'] = [r for r in duplicate_worker if r.t > 12000]

    for worker_records in by_worker.values():
        plot_time_line_data_per_worker(worker_records)
