systemname=""

cp ITPs.txt temp
grep -v "include" ${systemname}.top >> temp
mv temp ${systemname}.top

# Will append all necessary topology definitions to the COBY generated .top file