import copy,unittest
from stylelib.common import fingerprint
from stylelib.resume_policy import authorize_resume,LEGACY_TRAINER_FINGERPRINT

class ResumeSafetyTests(unittest.TestCase):
    def inputs(self):return {'records':'fixed-data','cache_settings':{'sources':'fixed-weights'},'contract':{'optimizer':{'learning_rate':1e-5}},'profile':'b200','offload':True,'seed':42,'purpose':'pilot','reference_fraction':0,'trainer_code':'current-code','effective_learning_rate':3e-5}
    def legacy(self,current):
        old={k:v for k,v in current.items() if k!='effective_learning_rate'};old['trainer_code']=LEGACY_TRAINER_FINGERPRINT
        return {'identity':fingerprint(old),'config':current['contract'],'step':20}
    def test_pinned_legacy_lr_upgrade_requires_explicit_flag(self):
        current=self.inputs();saved=self.legacy(current)
        with self.assertRaises(ValueError):authorize_resume(saved,current)
        result=authorize_resume(saved,current,True)
        self.assertEqual(result['step'],20);self.assertTrue(result['optimizer_and_rng_preserved'])
    def test_legacy_cannot_change_dataset_even_with_lr_permission(self):
        current=self.inputs();saved=self.legacy(current);current['records']='other-data'
        with self.assertRaises(ValueError):authorize_resume(saved,current,True)
    def test_new_checkpoint_cannot_change_profile_or_code(self):
        old=self.inputs();saved={'identity':fingerprint(old),'resume_identity_inputs':old,'step':25}
        for key in ('profile','trainer_code','seed'):
            new=copy.deepcopy(old);new[key]='changed'
            with self.assertRaises(ValueError):authorize_resume(saved,new,True)
    def test_new_checkpoint_same_inputs_resumes_without_special_permission(self):
        current=self.inputs();saved={'identity':fingerprint(current),'resume_identity_inputs':current,'step':25}
        self.assertIsNone(authorize_resume(saved,current))
