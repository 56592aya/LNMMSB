labels = read.csv("Dropbox/Arash/EUR/Workspace/CLSM/Julia Implementation/ProjectLN/data/SNAP_COMMS/Email/email-Eu-core-department-labels.txt", sep = " ")
edgelist = read.csv("Dropbox/Arash/EUR/Workspace/CLSM/Julia Implementation/ProjectLN/data/SNAP_COMMS/Email/email-Eu-core.txt", sep=" ")
edgelist = edgelist + 1
labels = labels + 1

