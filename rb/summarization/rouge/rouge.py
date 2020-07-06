from pathlib import Path
from pyrouge import Rouge155

import logging
logging.disable(logging.INFO)


class Rouge(object):
    
    _INSTANCE = None
    
    def __init__(self, relative_path: Path = Path.cwd()):
        self._r = Rouge155()
        
        self._r.system_dir = str(relative_path / "system_summaries")     # your summaries
        self._r.model_dir = str(relative_path / "model_summaries")       # gold standard summaries
        
        self._r.system_filename_pattern = r'duc2002.(\d+).txt'
        self._r.model_filename_pattern = r'duc2002.[A-Z].#ID#.txt'
        
    @classmethod
    def get_instance(cls) -> "Rouge":
        """
        Note: Multiple calls of evaluate method with the same instance will throw "Illegal division by zero" exception.
        :return: unique instance of @Rouge class
        """
        if cls._INSTANCE is None:
            cls._INSTANCE = Rouge()
        return cls._INSTANCE
    
    @property
    def system_dir(self):
        return self._r.system_dir

    @property
    def model_dir(self):
        return self._r.model_dir
    
    def evaluate(self):
        """
        Evaluate and returns the result as a dictionary.
        :return: dictionary with results
        """
        options = "-e /home/vacioaca/ROUGE-1.5.5/data -c 95 -r 1000 -n 4 -w 1.2 -a -x"
        output = self._r.convert_and_evaluate(rouge_args=options)
        return output
        # return self._r.output_to_dict(output)


if __name__ == "__main__":
    rouge = Rouge()

    result = rouge.evaluate()
    print(result)
