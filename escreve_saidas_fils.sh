#!/bin/csh -f
#escreve artigos com todas as informacoes:
# de fracao de azuis 
# de fracao de vermelhas
# de densidade, cmprimento, etc

lista_cls=/home/natalia/Dados/XMM_catalogues/cls.txt
file_out_blue=/home/natalia/Dados/filamentos/SCMS/BlueFrac.csv
# file_out_red=/home/natalia/Dados/filamentos/SCMS/RedFrac.csv'
file_out_densi=/home/natalia/Dados/filamentos/SCMS/DensiCompri.csv

rm -f $file_out_blue $file_out_red $file_out_densi
echo 'Cluster, Length (deg), Length (Mpc), Density, Relative Density' >$file_out_densi
echo 'Cluster, Blue_peak, Blue_peak_std, Red_peak, Red_peak_std, Gvalley, BF_Fil, BF_Fil_err, RF_campo, RF_campo_err ' >$file_out_blue


while IFS= read -r line
do
  ag=$(echo "$line")
  echo $ag

  path=/home/natalia/Dados/filamentos/SCMS/$ag
  cd $path

  # DENISDADE E COMPRIMENTO
  densi=$(cat f1_comrpimento.txt | tail -1)
  echo $ag', '$densi >>$file_out_densi

  #FRACAO DE CORES
  #FIL
  Blue_p=$(cat gi_campo_distrib_values.csv | tail -1 | cut -d ',' -f 2 | xargs)
  Blue_pEr=$(cat gi_campo_distrib_values.csv | tail -1 | cut -d ',' -f 3| xargs)
  Red_p=$(cat gi_campo_distrib_values.csv | tail -1 | cut -d ',' -f 4| xargs)
  Red_pEr=$(cat gi_campo_distrib_values.csv | tail -1 | cut -d ',' -f 5| xargs)
  corcampo=$(cat gi_campo_colorFrac.csv | tail -1)
  echo $ag', '$Blue_p', '$Blue_pEr', '$Red_p', ' $Red_pEr', '$corcampo>>$file_out_blue

  #FRACAO VERMELHAS POR RAIO | FIL


done < "$lista_cls"



