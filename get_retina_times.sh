mkdir -p retina-times-gpu

echo "BC,PC,CO,RD,UN,PG,HB,CA" >> retina-times-gpu/retina.eval.times
./autofuse3.sh retinamnist BC,PC,CO,RD,UN,PG,HB,CA | grep "Time taken to evaluate" >> retina-times-gpu/retina.eval.times

echo "BC,PC,CO,RD,UN,PG,HB" >> retina-times-gpu/retina.eval.times
./autofuse3.sh retinamnist BC,PC,CO,RD,UN,PG,HB | grep "Time taken to evaluate" >> retina-times-gpu/retina.eval.times

echo "BC,PC,CO,RD,UN,PG" >> retina-times-gpu/retina.eval.times
./autofuse3.sh retinamnist BC,PC,CO,RD,UN,PG | grep "Time taken to evaluate" >> retina-times-gpu/retina.eval.times

echo "BC,PC,CO,RD,UN" >> retina-times-gpu/retina.eval.times
./autofuse3.sh retinamnist BC,PC,CO,RD,UN | grep "Time taken to evaluate" >> retina-times-gpu/retina.eval.times

echo "BC,PC,CO,RD" >> retina-times-gpu/retina.eval.times
./autofuse3.sh retinamnist BC,PC,CO,RD | grep "Time taken to evaluate" >> retina-times-gpu/retina.eval.times

echo "BC,PC,CO" >> retina-times-gpu/retina.eval.times
./autofuse3.sh retinamnist BC,PC,CO | grep "Time taken to evaluate" >> retina-times-gpu/retina.eval.times

echo "BC,PC" >> retina-times-gpu/retina.eval.times
./autofuse3.sh retinamnist BC,PC | grep "Time taken to evaluate" >> retina-times-gpu/retina.eval.times

echo "BC" >> retina-times-gpu/retina.eval.times
./autofuse3.sh retinamnist BC | grep "Time taken to evaluate" >> retina-times-gpu/retina.eval.times
