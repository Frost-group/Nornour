systemname=""

# Will create an index file defining which particles belong to the 'Solvent' and 'Membrane' groups
# Useful for simulation conditions where system components need to be treated with separate conditions e.g. temp or pressure

echo "keep 0" > index-selection.txt
echo "r W | r NA | r CL" >> index-selection.txt
echo "name 1 Solvent" >> index-selection.txt
echo '!1' >> index-selection.txt
echo "name 2 Membrane" >> index-selection.txt
echo "q" >> index-selection.txt

gmx make_ndx -f ${systemname}.gro -o index.ndx < index-selection.txt
