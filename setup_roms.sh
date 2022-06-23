wget http://www.atarimania.com/roms/Roms.rar
apt install unrar
unrar e Roms.rar ROMS/
python3 -m atari_py.import_roms ROMS
