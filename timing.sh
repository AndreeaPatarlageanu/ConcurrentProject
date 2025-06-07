export TIMEFORMAT=%R

for N in 10000 100000 1000000 10000000 100000000 1000000000 10000000000; do
  awk -v n="$N" '/^>/{i++} i<=n{print}' SwissProt.fasta > queries_${N}.fa
  ./makedb SwissProt.fasta benchdb/sp
  echo -n "${N} sequences: "
  { time ./align --query queries_${N}.fa --db benchdb/sp; } 2>&1 | tail -n1
done
