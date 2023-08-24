#!/bin/sh

echo -n "Progress: "
for i in $(seq 1 100); do
    sleep 0.01  # Sleep for demonstration purposes
    echo -n '\r'"$i%"
done
echo

# Define a function
progress() {
    echo -n "Progress: "
    for i in $(seq 1 100); do
        # Sleep for demonstration purposes
        sleep 0.01
        # Overwrite output in-place
        printf "\rProgress: %d%%" "$i"
    done
    echo
}


# Run a command, display output in real-time, and capture output
output=$( progress 2>&1 | tee /dev/fd/2)

# Use the output
echo "----"
echo "Captured output:"
echo "$output"
