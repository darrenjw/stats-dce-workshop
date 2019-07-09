# Makefile


build:
	cd GraphicalModels ; make
	cd BayesianModelling ; make
	cd ComputerModels ; make

edit:
	emacs Makefile *.md &


# eof
