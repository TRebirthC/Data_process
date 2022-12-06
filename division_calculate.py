import numpy as np

dir = "rslt.division/"
tail = ".division.s"

def data_id_2name(project_id):
    """2021-12-19. the below projects suffer issues individually as below.
    homebrew        our method arises an error, no available data
    neutron         ood, error happens at the 10,000 steps
    npm             oob, error n_data < 10,000
    spring-integration, <10,000
    """
    if project_id == 0:
        project_name = "brackets"
    elif project_id == 1:
        project_name = "broadleaf"
    elif project_id == 2:
        project_name = "camel"
    elif project_id == 3:
        project_name = "corefx"
    elif project_id == 4:
        project_name = "django"
    elif project_id == 5:
        project_name = "fabric"
    elif project_id == 6:
        project_name = "jgroups"
    elif project_id == 7:
        project_name = "nova"
    elif project_id == 8:
        project_name = "rails"
    elif project_id == 9:
        project_name = "rust"
    elif project_id == 10:
        project_name = "tensorflow"
    elif project_id == 11:
        project_name = "tomcat"
    elif project_id == 12:
        project_name = "vscode"
    elif project_id == 13:
        project_name = "wp-calypso"
    else:
        raise Exception("undefined data id.")
    return project_name

for i in range(14):
    data = data_id_2name(i)
    now_dir = dir + data + "/"
    odasc_dir = now_dir + "odasc" + tail
    oob_dir = now_dir + "oob" + tail
    odasc_result = np.loadtxt(odasc_dir)
    odasc_i = 0
    odasc_d = 0
    if odasc_result.size > 1:
        for j in range(len(odasc_result)):
            if j == 0:
                odasc_i = odasc_result[j]
            elif j%2 == 1:
                odasc_d = odasc_d + (odasc_result[j+1] - odasc_result[j])
    else:
        odasc_i = odasc_result
    print("odasc in " + data + ": " + str(odasc_i) + " " + str(odasc_d))

    oob_result = np.loadtxt(oob_dir)
    oob_i = 0
    oob_d = 0
    if oob_result.size > 1:
        for j in range(len(oob_result)):
            if j == 0:
                oob_i = oob_result[j]
            elif j % 2 == 1:
                oob_d = oob_d + (oob_result[j + 1] - oob_result[j])
    else:
        oob_i = oob_result
    print("oob in " + data + ": " + str(oob_i) + " " + str(oob_d))

for i in range(14):
    data = data_id_2name(i)
    now_dir = dir + data + "/"
    pbsa_dir = now_dir + "pbsa" + tail
    pbsa_result = np.loadtxt(pbsa_dir)
    pbsa_i = 0
    pbsa_d = 0
    if pbsa_result.size > 1:
        for j in range(len(pbsa_result)):
            if j == 0:
                pbsa_i = pbsa_result[j]
            elif j%2 == 1:
                pbsa_d = pbsa_d + (pbsa_result[j+1] - pbsa_result[j])
    else:
        pbsa_i = pbsa_result
    print("pbsa in " + data + ": " + str(pbsa_i) + " " + str(pbsa_d))
