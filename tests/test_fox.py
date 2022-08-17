import sys, os
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../fox')

from foxplainer.explainer import FoX
import pytest

@pytest.mark.parametrize("""global_model_name, 
                            xtype,
                            xnum,
                            global_model_path,
                            proj_name,
                            data_path,
                            inst_id""",
                         [
                             ("LR", "abd", 1, "./foxplainer/global_model/openstack_LR_global_model.pkl", "openstack", "./foxplainer/dataset/", 5),
                             ("RF", "abd", 1, "./foxplainer/global_model/openstack_RF_30estimators_global_model.pkl", "openstack", "./foxplainer/dataset/", 5),
                             ("LR", "con", 1, "./foxplainer/global_model/openstack_LR_global_model.pkl", "openstack", "./foxplainer/dataset/", 5),
                             ("RF", "con", 1, "./foxplainer/global_model/openstack_RF_30estimators_global_model.pkl", "openstack", "./foxplainer/dataset/", 5),
                             ("LR", "con", "all", "./foxplainer/global_model/openstack_LR_global_model.pkl", "openstack", "./foxplainer/dataset/", 5),
                             ("RF", "con", "all", "./foxplainer/global_model/openstack_RF_30estimators_global_model.pkl", "openstack", "./foxplainer/dataset/", 5)                         
                         ])
def test_exp(global_model_name, xtype, xnum, global_model_path, proj_name, data_path, inst_id):
    fx = FoX(global_model_name=global_model_name, 
            xtype=xtype, 
            xnum=xnum, 
            global_model_path=global_model_path, 
            proj_name=proj_name, 
            data_path=data_path,
            inst_id=inst_id)
    fx.explain(in_jupyter=True)
    fx.explain(in_jupyter=False)
