NRUNS=9

cd Resized_Images

for ((i=0;i<=NRUNS;i++)); do
    mkdir 0000$i
done

NRUNS=42
for ((i=10;i<=NRUNS;i++)); do
    mkdir 000$i
done
