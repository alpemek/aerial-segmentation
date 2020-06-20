mkdir -p dataset
cd dataset
wget https://zenodo.org/record/1154821/files/berlin.zip
unzip berlin.zip && rm berlin.zip
wget https://zenodo.org/record/1154821/files/chicago.zip
unzip chicago.zip &&  rm chicago.zip
wget https://zenodo.org/record/1154821/files/paris.zip
unzip paris.zip && rm paris.zip
wget https://zenodo.org/record/1154821/files/zurich.zip
unzip zurich.zip && rm zurich.zip
cd ~