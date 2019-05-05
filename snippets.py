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
    """Evaluates the latency score based on the `Pixel Switching Times` (pst)."""
    
    # Check if there is a head movement
    movement = self.check_for_head_movement(stream=head_stream)

    # If there is no movement, returns 0.
    if not movement:
        return 0

 """Snippet 7"""
    

 def generate_final_score(self, scores):
    """Generators the final latency score."""
    
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
        
        # prepare the frame for processing
        frame = prep_image(frame)
        frame = resize_image(frame, width=400)
        (h, w) = frame.shape[:2
                             
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        self.face_detector.setInput(blob)
        detections = self.face_detector.forward()

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence
            confidence = detections[0, 0, i, 2]

            # filter out weak detections
            if confidence < 0.5:
                continue

            # compute the (x, y)-coordinates of the bounding box
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # extract the face ROI
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face, (64, 64))
            face = face.astype("float") / 255.0
            face = img_to_array(face)
            face = np.expand_dims(face, axis=0)

            predictions = self.emotion_detector.predict(face)[0]
            label = self.POSSIBLE_EMOTIONS[predictions.argmax()]
            
            return label

    """ Snippet 9 """"
    
    def generate_final_score(self, all_emotions, expected_emotions):
    """Generators the final sentiment score."""
                             
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
        """Generates the final questionnaire score."""
                             
        # Calculate the pre and post questionnaire scores.
        _, _, _, pre_ts = self._calculate_pre_score(pre=pre)
        _, _, _, post_ts = self._calculate_post_score(post=post)

        # Calculating the total questionnaire score.
        tq = ((post_ts - pre_ts) / self.QUESTIONNAIRE_MAX_TOTAL_SCORE) * 100

        # check if score is less than 0, if yes, moderate it to 0
        if tq < 0:
            tq = 0

        return tq


"""Snippet 11"""

    def generate_cssi_score(self, tl, ts, tq, plugin_scores=None):
        """Generators the final CSSI score."""
                             
        tot_ps = 0.0  # Variable to store the sum of the plugin scores
        tot_pw = 0  # Variable to keep track total plugin weight

        # Checks if any plugins are provided for score calculation.
        if plugin_scores is not None:
            for plugin in plugin_scores:
                plugin_name = plugin["name"]
                # Checks if the plugin is registered in the configuration file
                # If not, raises an exception.
                if plugin_name not in self.config.plugins:
                    raise CSSIException("The plugin {0} appears to be invalid.".format(plugin_name))
                else:
                    plugin_weight = float(self.config.plugin_options[plugin_name]["weight"]) / 100
                    plugin_score = plugin["score"]

                    # Checks if the passed in plugin score is less than 100.
                    # If not an exception will be thrown.
                    if plugin_score > 100:
                        raise CSSIException("Invalid score provided for the plugin: {0}.".format(plugin_name))

                    # Ads the current plugin score to the total plugin score.
                    tot_ps += plugin_score * plugin_weight
                    # Ads the current plugin weight to the total plugin weight percentage.
                    tot_pw += plugin_weight

        lw = float(self.config.latency_weight) / 100  # latency weight percentage
        sw = float(self.config.sentiment_weight) / 100  # sentiment weight percentage
        qw = float(self.config.questionnaire_weight) / 100  # questionnaire weight percentage

        # Checks if the total weight is less than 100 percent.
        if (lw + sw + qw + tot_pw) > 1:
            raise CSSIException("Invalid weight configuration. Please reconfigure and try again")

        # Calculating the CSSI score
        cssi = (tl * lw) + (ts * sw) + (tq * qw) + tot_ps

        # Double checks if the generated CSSI score is less than 100.
        if cssi > 100:
            raise CSSIException("Invalid CSSI score was generated. Please try again")

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
                             
"""Snippet 13"""

    def _calculate_ssq_total_score(self, questionnaire, filename):
        """Calculates the SSQ score for a particular questionnaire"""
        _n = 0.0
        _o = 0.0
        _d = 0.0
        try:
            with open(self._get_meta_file_path(filename)) as meta_file:
                meta = json.load(meta_file)
                # Iterate through the symptoms and generate the
                # populate the `N`, `O` & `D` symptom scores.
                for s in meta["symptoms"]:
                    if s["weight"]["N"] == 1:
                        _n += int(questionnaire[s["symptom"]])
                    if s["weight"]["O"] == 1:
                        _o += int(questionnaire[s["symptom"]])
                    if s["weight"]["D"] == 1:
                        _d += int(questionnaire[s["symptom"]])

                # Calculate the `N`, `O` & `D` weighted scores.
                # and finally compute the total score.
                n = _n * meta["conversion_multipliers"]["N"]
                o = _o * meta["conversion_multipliers"]["O"]
                d = _d * meta["conversion_multipliers"]["D"]
                ts = (_n + _o + _d) * meta["conversion_multipliers"]["TS"]

                return n, o, d, ts
        except FileNotFoundError as error:
            raise CSSIException(
                "Questionnaire meta file couldn't not be found at %s" % (self._get_meta_file_path(filename))
            ) from error

    @staticmethod
    def _get_meta_file_path(filename):
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "meta", filename)     
                             
"""Snippet 14"""
                             
                             
@celery.task
def calculate_latency(session_id, limit):
    """Celery task which handles latency score generation and persistence"""
    from .wsgi_aux import app
    with app.app_context():
        head_key = "head-frames"
        scene_key = "scene-frames"

        r = redis.StrictRedis(host='localhost', port=6379, db=0)
        head_frames_raw = get_frames_from_redis(r=r, key=head_key, limit=limit)
        scene_frames_raw = get_frames_from_redis(r=r, key=scene_key, limit=limit)

        head_stream = []
        scene_stream = []

        for data in head_frames_raw:
            head_stream.append(decode_base64(data))

        for data in scene_frames_raw:
            scene_stream.append(decode_base64(data))

        _, phf_pitch, phf_yaw, phf_roll = cssi.latency.calculate_head_pose(frame=head_stream[0])
        _, chf_pitch, chf_yaw, chf_roll = cssi.latency.calculate_head_pose(frame=head_stream[1])
        _, _, ff_angles, sf_angles = cssi.latency.calculate_camera_pose(first_frame=scene_stream[0],
                                                                        second_frame=scene_stream[1], crop=True,
                                                                        crop_direction='horizontal')

        head_angles = [[phf_pitch, phf_yaw, phf_roll], [chf_pitch, chf_yaw, chf_roll]]
        camera_angles = [ff_angles, sf_angles]

        latency_score = cssi.latency.generate_rotation_latency_score(head_angles=head_angles,
                                                                     camera_angles=camera_angles)

        head_movement = cssi.latency.check_for_head_movement(head_stream)
        logger.debug("Head movement detected: {0}".format(head_movement))

        pst = cssi.latency.calculate_pst(scene_stream, 10)
        logger.debug("Pixel switching time: {0}".format(pst))

        session = Session.query.filter_by(id=session_id).first()
        if session is not None:
            new_score = {'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'score': latency_score}
            session.latency_scores.append(new_score)
            db.session.commit()
                             
                             
  
print('')
