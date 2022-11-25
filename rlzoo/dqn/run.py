from collections import OrderedDict as odcit
import torzilla as tz

A = tz.Argument

_BATCH_SIZE = 1024

CONFIG = odcit(
    num_worker = A(type=int, default=10),
    num_learner = A(type=int, default=8),
    num_replay_buffer = A(type=int, default=3),

    worker = {

    },

    learner = {

    },
    
    replay_buffer = odcit(
        capacity = A(type=int, default=_BATCH_SIZE*3),
        max_cache_size = A(type=int, default=_BATCH_SIZE*3),
    ),
)



def main():
    args = tz.parse_hargs(CONFIG)
    print(args)

if __name__ == '__main__':
    main()