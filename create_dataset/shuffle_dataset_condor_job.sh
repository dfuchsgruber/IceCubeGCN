universe = vanilla
log = /home/dfuchsgruber/log/shuffle_dataset.log
error = /home/dfuchsgruber/log/shuffle_dataset.err
output = /home/dfuchsgruber/log/shuffle_dataset.out

executable = create_dataset/shuffle_dataset.sh
arguments =

should_transfer_files = YES
when_to_transfer_output = ON_EXIT

request_cpus = 1
request_memory = 64GB

queue


