"""Snippet 1"""

from cssi.core import CSSI

# Create an instance of the CSSI Library
cssi = CSSI(
    shape_predictor="classifiers/hmd_face_landmarks.dat",
    debug=False,
    config_file="config.cssi"
)



"""Snippet 2"""

class CSSI(object):
    """The main access point for the CSSI library"""

    def __init__(self, shape_predictor, debug=False, config_file=None):
        """Initializes all the core modules in the CSSI Library."""
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
        logger.debug("CSSI library initialized......")

       

"""Snippet 3"""


def generate_rotation_latency_score(self, head_angles, camera_angles):
    """Evaluates the latency score for a corresponding head and scene rotation pair."""
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


"""Snippet 4"""


def check_for_head_movement(self, stream):
    """Checks whether if there is a head movement in a stream of head frames."""
    phf_pitch, phf_yaw, phf_roll = 0.0, 0.0, 0.0
    for idx, frame in enumerate(stream):
        _, chf_pitch, chf_yaw, chf_roll = self.calculate_head_pose(
            frame=frame)
        if idx != 0:
            if abs(calculate_angle_diff(angle_1=phf_pitch, angle_2=chf_pitch)) > 0:
                return True
            elif abs(calculate_angle_diff(angle_1=phf_yaw, angle_2=chf_yaw)) > 0:
                return True
            elif abs(calculate_angle_diff(angle_1=phf_roll, angle_2=chf_roll)) > 0:
                return True
        phf_pitch, phf_yaw, phf_roll = chf_pitch, chf_yaw, chf_roll
    return False

"""Snippet 5"""

@staticmethod
def calculate_pst(stream, fps):
    """Calculates the `Pixel Switching Times` (pst) of a camera frame stream."""
    prev_frame = None
    processed_count = 0
    equal_count = 0
    for idx, frame in enumerate(stream):
        processed_count += 1
        if idx != 0:
            diff = cv2.subtract(prev_frame, frame)
            B, G, R = cv2.split(diff)
            # If all the pixels (Red, Green &  Blue) are equal then the two images are similar.
            # If not then the images are different and the pst is calculated.
            if cv2.countNonZero(B) == 0 and cv2.countNonZero(G) == 0 and cv2.countNonZero(R) == 0:
                equal_count += 1
            else:
                return (processed_count / fps) * 1000
        prev_frame = frame
    # If the stream did not have any different frames, `None` will be returned.
    return None

"""Snippet 6"""

def generate_pst_latency_score(self, head_stream, camera_stream):
    """Evaluates the latency score based on the `Pixel Switching Times` (pst).

    This function first check if there is a head movement in the passed in head frame stream
    and if there is, it calculates the `Pixel Switching Times` (pst) of frames in the camera
    frame stream.

    Args:
        head_stream (list): List of head frames.
        camera_stream(list): List of camera frames(scene frames).

    Returns:
        int: If the pst is more than the motion-to-photon latency boundary which is specified
            in the configuration (default 20ms), a score of 1 will be returned. If there is no head
            movement or if the pst is less than the boundary, 0 will be returned.

    Examples:
        >>> cssi.latency.generate_pst_latency_score(head_stream, camera_stream)
    """
    # Check if there is a head movement
    movement = self.check_for_head_movement(stream=head_stream)

    # If there is no movement, returns 0.
    if not movement:
        return 0

 """Snippet 7"""
    

 def generate_final_score(self, scores):
    """Generators the final latency score.

    `sum_ln` is used to persist the sum of the individual latency scores.
    Then the sum is divided by n`, which is the number of latency tests carried out.
    The result is then multiplied by 100 to generate `tl` (Total Latency Score).

    Args:
        scores (list): A list of python dictionaries containing all the individual
            latency scores. ex: [{"score": 0,"timestamp": "2019-04-24 18:29:25"}]

    Returns:
        float: The total latency score.

    Examples:
        >>> cssi.latency.generate_final_score(scores)
    """
    n = len(scores)  # Total number of emotions captured
    sum_ls = 0.0  # Variable to store thr sum of the individual latency scores

    # Calculates the sum of latency scores.
    for score in scores:
        sum_ls += score['score']

    # Calculating the total latency score i.e `tl`
    tl = (sum_ls / n) * 100
    return tl

 """Snippet 8"""

def detect_emotions(self, frame):
    """Detects the sentiment on a face."""
    frame_resized = resize_image(frame, width=300)
    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=2, minSize=(30, 30),
                                                flags=cv2.CASCADE_SCALE_IMAGE)

    if len(faces) > 0:
        logger.debug("Number of Faces: {0}".format(len(faces)))
        faces = sorted(faces, reverse=True,
                       key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fx, fy, fw, fh) = faces

        # Extract the ROI of the face and resize it to 28x28 pixels
        # to make it compatible with the detector model.
        roi = gray[fy:fy + fh, fx:fx + fw]
        roi = cv2.resize(roi, (64, 64))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        predictions = self.emotion_detector.predict(roi)[0]
        label = self.POSSIBLE_EMOTIONS[predictions.argmax()]
        logger.debug("Identified emotion is: {0}".format(label))
        return label

    """ Snippet 9 """"
    
    def generate_final_score(self, all_emotions, expected_emotions):
    """Generators the final sentiment score.

    Different applications will cause the user to portray different emotions.
    ["angry", "disgust", "scared", "sad"] are considered negative emotions by default
    unless specified in the `expected_emotions` array.

    Args:
        all_emotions (list): A list of all the captured emotions
        expected_emotions (list): A list of expected emotions during the session.
    Returns:
        float: The total sentiment score.

    Examples:
        >>> cssi.sentiment.generate_final_score(all_emotions, expected_emotions)
    """
    n_tot = len(all_emotions)  # Total number of emotions captured
    n_neg = 0  # Variable to record the negative emotion count.

    # Checks if the emotion is negative and if it is, and if it is not in
    # the expected emotions list, `n_neg` will be incremented by one.
    for emotion in all_emotions:
        if emotion["sentiment"] in self.NEGATIVE_EMOTIONS:
            if emotion["sentiment"] not in expected_emotions:
                n_neg += 1

    # Calculating the total sentiment score.
    ts = (n_neg / n_tot) * 100
    return ts

    """ Snippet 10 """"
    
def generate_final_score(self, pre, post):
    """Generates the final questionnaire score.
    Args:
        pre (dict): Pre questionnaire results.
        post (dict): Post questionnaire results.
    Returns:
        float: The total questionnaire score.
    Examples:
        >>> cssi.questionnaire.generate_final_score(pre, post)
    """
    # Calculate the pre and post questionnaire scores.
    pre_n, pre_o, pre_d, pre_ts = self._calculate_pre_score(pre=pre)
    post_n, post_o, post_d, post_ts = self._calculate_post_score(post=post)

    # Calculating the total questionnaire score.
    tq = ((post_ts - pre_ts) / self.QUESTIONNAIRE_MAX_TOTAL_SCORE) * 100
    return tq


"""Snippet 11"""

def generate_cssi_score(self, tl, ts, tq, plugin_scores=None):
    """Generators the final CSSI score.

    This is the core function of the CSSI library and it takes in the scores and
    generates the final CSSI score for the test session.

    Args:
        tl (float): Total latency score
        ts (float): Total sentiment score
        tq (float): Total questionnaire score
        plugin_scores (list): A list of dictionaries containing plugin details.
            ex: [{"name": "heartrate.plugin", "score": 40.00}].
    Returns:
        float: The CSSI score.
    Raises:
        CSSIException: If the calculations couldn't be completed successfully
            this exception will be thrown.
    Examples:
        >>> cssi.generate_cssi_score(tl, ts, tq, plugin_scores)
    """
    tot_ps = 0.0  # Variable to store the sum of the plugin scores
    tot_pw = 0  # Variable to keep track total plugin weight

    # Checks if any plugins are provided for score calculation.
    if plugin_scores is not None:
        for plugin in plugin_scores:
            plugin_name = plugin["name"]
            # Checks if the plugin is registered in the configuration file
            # If not, raises an exception.
            if plugin_name not in self.config.plugins:
                raise CSSIException(
                    "The plugin {0} appears to be invalid.".format(plugin_name))
            else:
                plugin_weight = float(
                    self.config.plugin_options[plugin_name]["weight"]) / 100
                plugin_score = plugin["score"]

                # Checks if the passed in plugin score is less than 100.
                # If not an exception will be thrown.
                if plugin_score > 100:
                    raise CSSIException(
                        "Invalid score provided for the plugin: {0}.".format(plugin_name))

                # Ads the current plugin score to the total plugin score.
                tot_ps += plugin_score * plugin_weight
                # Ads the current plugin weight to the total plugin weight percentage.
                tot_pw += plugin_weight

    lw = float(self.config.latency_weight) / \
        100  # latency weight percentage
    sw = float(self.config.sentiment_weight) / \
        100  # sentiment weight percentage
    qw = float(self.config.questionnaire_weight) / \
        100  # questionnaire weight percentage

    # Checks if the total weight is less than 100 percent.
    if (lw + sw + qw + tot_pw) > 1:
        raise CSSIException(
            "Invalid weight configuration. Please reconfigure and try again")

    # Calculating the CSSI score
    cssi = (tl * lw) + (ts * sw) + (tq * qw) + tot_ps

    # Double checks if the generated CSSI score is less than 100.
    if cssi > 100:
        raise CSSIException(
            "Invalid CSSI score was generated. Please try again")

    return cssi


"""Snippet 12"""


@celery.task
def record_sentiment(head_frame, session_id):
    """Celery task which handles sentiment score generation and persistence"""
    from .wsgi_aux import app
    with app.app_context():
        sentiment = cssi.sentiment.detect_emotions(frame=head_frame)
        session = Session.query.filter_by(id=session_id).first()
        if session is not None:
            if sentiment is not None:
                new_score = {'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'sentiment': sentiment}
                session.sentiment_scores.append(new_score)
                db.session.commit()
                

print('')
