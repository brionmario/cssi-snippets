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


    def generate_rotation_latency_score(self, head_angles, camera_angles):
        """Evaluates the latency score for a corresponding head and scene rotation pair.

        Args:
            head_angles (list): Pair of head rotation angles in a list. i.e pitch, yaw and
                roll of previous and current rotation.
            camera_angles(list): Pair of scene rotation angles in a list. i.e pitch, yaw and
                roll of previous and current rotation.

        Returns:
            int: If there is a discrepancy in the rotations 1 will be returned, else 0.

        Examples:
            >>> cssi.latency.generate_rotation_latency_score(head_angles, camera_angles)
        """
        # Calculates the difference of the angle pairs.
        hp_diff, hy_diff, hr_diff = self._calculate_angle_pair_diff(head_angles)
        cp_diff, cy_diff, cr_diff = self._calculate_angle_pair_diff(camera_angles)

        # Checks if the difference between the angles is greater than the measurement error.
        # If yes for either pitch, yaw or roll difference, 1 will be returned, else 0.
        if abs(calculate_angle_diff(hp_diff, cp_diff)) >= self.ANGLE_MEASUREMENT_ERR:
            return 1
        elif abs(calculate_angle_diff(hy_diff, cy_diff)) >= self.ANGLE_MEASUREMENT_ERR:
            return 1
        elif abs(calculate_angle_diff(hr_diff, cr_diff)) >= self.ANGLE_MEASUREMENT_ERR:
            return 1
        return 0
