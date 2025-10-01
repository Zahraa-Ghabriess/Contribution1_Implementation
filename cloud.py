# The structure of the server
# The server should include the following functions:
# 1. Server initialization
# 2. Server reveives updates from the user
# 3. Server send the aggregated information back to clients
import copy
from average import average_weights

class Cloud():

    def __init__(self, shared_layers):
        self.receiver_buffer = {}
        self.id_registration = []
        self.sample_registration = {}
        self.shared_state_dict = self.get_model_parameters(shared_layers)
        self.clock = []

    def get_model_parameters(self, model):
        """Returns the parameters of a sklearn model."""
        if model.fit_intercept:
            params = {
                'coef': model.coef_,
                'intercept': model.intercept_, # bias
            }
        else:
            params = {
                'coef': model.coef_,
            }
        return params
    
    def refresh_cloudserver(self):
        self.receiver_buffer.clear()
        del self.id_registration[:]
        self.sample_registration.clear()
        return None

    def edge_register(self, edge):
        self.id_registration.append(edge.id)
        self.sample_registration[edge.id] = edge.all_trainsample_num
        return None

    def receive_from_edge(self, edge_id, eshared_state_dict):
        self.receiver_buffer[edge_id] = eshared_state_dict
        return None

    def aggregate(self, args):
        received_dict = [dict for dict in self.receiver_buffer.values()]
        sample_num = [snum for snum in self.sample_registration.values()]
        self.shared_state_dict = average_weights(w=received_dict,
                                                 s_num=sample_num)
        return None

    def send_to_edge(self, edge):
        edge.receive_from_cloudserver(copy.deepcopy(self.shared_state_dict))
        return None

