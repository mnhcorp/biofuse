# For all datasets: retinamnist, pneumoniamnist, pathmnist, octmnist, dermamnist, breastmnist, organamnist, organsmnist, organcmnist, bloodmnist, tissuemnist, chestmnist
mkdir -p all-times-gpu

echo "Running autofuse3.sh for retinamnist"
echo "RetinaMNIST" >> all-times-gpu/eval.times
./autofuse3.sh retinamnist BC | grep "Time taken to evaluate" >> all-times-gpu/eval.times

echo "Running autofuse3.sh for pneumoniamnist"
echo "PneumoniaMNIST" >> all-times-gpu/eval.times
./autofuse3.sh pneumoniamnist BC | grep "Time taken to evaluate" >> all-times-gpu/eval.times

echo "Running autofuse3.sh for pathmnist"
echo "PathMNIST" >> all-times-gpu/eval.times
./autofuse3.sh pathmnist BC | grep "Time taken to evaluate" >> all-times-gpu/eval.times

echo "Running autofuse3.sh for octmnist"
echo "OCTMNIST" >> all-times-gpu/eval.times
./autofuse3.sh octmnist BC | grep "Time taken to evaluate" >> all-times-gpu/eval.times

echo "Running autofuse3.sh for dermamnist"
echo "DermAMNIST" >> all-times-gpu/eval.times
./autofuse3.sh dermamnist BC | grep "Time taken to evaluate" >> all-times-gpu/eval.times

echo "Running autofuse3.sh for breastmnist"
echo "BreastMNIST" >> all-times-gpu/eval.times
./autofuse3.sh breastmnist BC | grep "Time taken to evaluate" >> all-times-gpu/eval.times

echo "Running autofuse3.sh for organamnist"
echo "OrganAMNIST" >> all-times-gpu/eval.times
./autofuse3.sh organamnist BC | grep "Time taken to evaluate" >> all-times-gpu/eval.times

echo "Running autofuse3.sh for organsmnist"
echo "OrgansMNIST" >> all-times-gpu/eval.times
./autofuse3.sh organsmnist BC | grep "Time taken to evaluate" >> all-times-gpu/eval.times

echo "Running autofuse3.sh for organcmnist"
echo "OrganCMNIST" >> all-times-gpu/eval.times
./autofuse3.sh organcmnist BC | grep "Time taken to evaluate" >> all-times-gpu/eval.times

echo "Running autofuse3.sh for bloodmnist"
echo "BloodMNIST" >> all-times-gpu/eval.times
./autofuse3.sh bloodmnist BC | grep "Time taken to evaluate" >> all-times-gpu/eval.times

echo "Running autofuse3.sh for tissuemnist"
echo "TissueMNIST" >> all-times-gpu/eval.times
./autofuse3.sh tissuemnist BC | grep "Time taken to evaluate" >> all-times-gpu/eval.times

echo "Running autofuse3.sh for chestmnist"
echo "ChestMNIST" >> all-times-gpu/eval.times
./autofuse3.sh chestmnist BC | grep "Time taken to evaluate" >> all-times-gpu/eval.times