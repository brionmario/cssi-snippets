"""Snippet 1"""

from cssi.core import CSSI

# Create an instance of the CSSI Library
cssi = CSSI(shape_predictor="classifiers/shape_predictor_68_face_landmarks.dat", debug=False, config_file="cssi.rc")



"""Snippet 2"""

class CSSI(object):
    """The main access point for the CSSI library"""

    def __init__(self, shape_predictor, debug=False, config_file=None):
        """ Initializes all the core modules in the CSSI Library.

        :param shape_predictor: Path to the landmark detector.
        :param debug: Boolean indicating if debug mode should be activated or not.
        :param config_file: A file containing all the configurations for CSSI.
        """
        # If no config file name is passed in, defaults to `config.cssi`
        if config_file is None:
            self.config_file = "config.cssi"
        self.config_file = config_file
        # Sets the debug mode
        self.debug = debug
        # Tries to read the config file.
        self.config = read_cssi_config(filename=self.config_file)
        # Initialize the latency capturing module
        self.latency = Latency(config=self.config, debug=self.debug, shape_predictor=shape_predictor)
        # Initializing the Sentiment capturing module
        self.sentiment = Sentiment(config=self.config, debug=self.debug)
        # Initializing the questionnaire module.
        self.questionnaire = SSQ(config=self.config, debug=self.debug)
        # Initializing the plugins specified in the configuration file
        self.plugins = Plugins.init_plugins(modules=self.config.plugins, config=self.config, debug=self.debug)
        logger.debug("CSSI library initialized......")

        
print('end')
