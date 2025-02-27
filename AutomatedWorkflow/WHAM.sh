# re-running WHAM as file system sync issues?

module load GROMACS/2024.1-foss-2023b

for peptide in RRWKIVVIRWRR RRWRIVVIRVR  RWWKIWVIRWWR
do
	for membrane in DOPC_CHOL DOPE_DOPG
	do
		cd "${peptide}_${membrane}"
pwd

    rm ${peptide}_${membrane}_tpr-files.dat ${peptide}_${membrane}_pullx-files.dat
    for i in {0..127}
    do
    	echo "${peptide}_${membrane}_umbrella${i}.tpr" >> ${peptide}_${membrane}_tpr-files.dat
    	echo "${peptide}_${membrane}_umbrella${i}_pullx.xvg" >> ${peptide}_${membrane}_pullx-files.dat
    done

		gmx wham -it ${peptide}_${membrane}_tpr-files.dat -ix ${peptide}_${membrane}_pullx-files.dat -o ${peptide}_${membrane}_pmf.xvg -hist ${peptide}_${membrane}_hist.xvg -unit kCal
cd -

done
done

