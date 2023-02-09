from pycls.elastic.dynamic_net import DynamicRegNet


class RawArch2IRConverter:
    def __init__(self, arch_manager):
        self.arch_manager = arch_manager

    def _get_arch(self, sample):
        arch = self.arch_manager.random_sample(based_raw_arch=sample, retry=False)
        assert arch is not None
        return arch
    
    def convert(self, sample):
        arch = self._get_arch(sample)
        return DynamicRegNet.gen_ir(**arch)

    def get_flops(self, sample):
        arch = self._get_arch(sample)
        cx = DynamicRegNet.complexity(**arch)
        return cx["flops"]
