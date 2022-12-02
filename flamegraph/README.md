# installation
sudo apt-get install linux-tools-common linux-tools-generic linux-tools-`uname -r`
curl -SO https://raw.githubusercontent.com/brendangregg/FlameGraph/master/stackcollapse-perf.pl
curl -SO https://raw.githubusercontent.com/brendangregg/FlameGraph/master/flamegraph.pl


perf record -F 99 -p 181 -g -- sleep 60

perf script > out.perf

https://raw.githubusercontent.com/brendangregg/FlameGraph/master/stackcollapse-perf.pl
stackcollapse-perf.pl

./stackcollapse-perf.pl out.perf > out.folded

./flamegraph.pl out.kern_folded > kernel.svg
#https://raw.githubusercontent.com/brendangregg/FlameGraph/master/flamegraph.pl

# note grep cpuid out.kern_folded | ./flamegraph.pl > cpuid.svg
