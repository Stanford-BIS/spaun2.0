import nengo
import nengo_brainstorm_pp.preprocessing as pp

import numpy as np

class Stack:
     def __init__(self):
         self.items = []

     def isEmpty(self):
         return self.items == []

     def push(self, item):
         self.items.append(item)

     def pop(self):
         return self.items.pop()

     def peek(self):
         return self.items[len(self.items)-1]

     def size(self):
         return len(self.items)


def calc_new_costs(model):                                                      
    encoding_weights = 0                                                        
    decoding_weights = 0                                                        
    transform_count = 0                                                         
    for conn in model.all_connections:                                          
        if isinstance(conn.pre_obj, nengo.Ensemble):                            
            assert isinstance(conn.post_obj, pp.Decoder)                            
            # cost for decoder                                                     
            ens = conn.pre_obj                                                  
            decoder = conn.post_obj                                             
            decoding_weights += ens.n_neurons * decoder.size_in                     
        elif isinstance(conn.pre_obj, pp.Encoder):                                  
            assert isinstance(conn.post_obj, nengo.Ensemble)                    
            ens = conn.post_obj                                                 
            encoding_weights += ens.dimensions * ens.n_neurons                  
        elif isinstance(conn.post_obj, nengo.ensemble.Neurons):                 
            if isinstance(conn.pre_obj, pp.NeuronEncoder):                          
                assert conn.pre_obj.size_out == 1                               
                ens = conn.post_obj.ensemble                                    
                encoding_weights += ens.dimensions * ens.n_neurons              
                                                                                
        if isinstance(conn.pre_obj, nengo.Node) and isinstance(conn.post_obj, nengo.Node) \
            and not pp.is_identity_like(conn.transform):                        
            transform_count += (conn.pre_obj.size_out * conn.post_obj.size_in)   
                                                                                
                                                                                
    return encoding_weights, decoding_weights, transform_count                  

def compute_stats(model,print_results=False):
    # Number of ensembles with no output connection; This should never happen          
    num_ensembles_with_no_output_conn = 0                                           
    # Dictionary mapping Ensemble to it's total synapse count                       
    num_synapses_per_ens = dict()                                                   
    # Dictionary mapping Ensemble to the Ensembles it's output is directed to          
    ens_ens_conn = dict()                                                           
    # Temporary variable to walk the path from an Ensemble and all Ensembles who are recieving input from the Ensemble.
    conn_stack = Stack()                                                            
    # Dictionary mapping Ensemble to list of Connections being fed by Ensemble         
    adjacency_list = dict()                                                         
                                                                                
    #############################################################################   
    # Create an adjacency list to make it easier to perform                         
    # single-source-multiple-destination walks through the network                  
    #############################################################################   

    for conn in model.all_connections:                                    
        if conn.pre_obj in adjacency_list:                                          
            adjacency_list[conn.pre_obj].append(conn)                               
        else:                                                                       
            adjacency_list[conn.pre_obj] = [conn]                                   

    #############################################################################   
    # Populate the ens_ens_conn dictionary                                          
    #############################################################################   
    for ens in model.all_ensembles:                                         
        ens_ens_conn[ens] = []                                                         
        if ens not in adjacency_list:                                                  
            num_ensembles_with_no_output_conn += 1                                     
            #print ens," has no output connections"                                    
        else:                                                                          
            # Now visit this Ensembles children, storing Ensembles that this Ensemble is connected to

            # First, add an initial set of Connections to the stack                 
            for conn in adjacency_list[ens]:                                        
                conn_stack.push(conn)                                               

            # Now while there still Connections to visit                               
            while not(conn_stack.isEmpty()):                                           
                # Pop the top connection                                               
                top_conn = conn_stack.pop()                                            
                # If the child of the top connection is an ensemble, add it to the dictionary and this path is done
                if isinstance(top_conn.post_obj,nengo.Ensemble):                       
                    if top_conn.post_obj not in ens_ens_conn[ens]:                     
                        ens_ens_conn[ens].append(top_conn.post_obj)                    
                else:                                                                  
                    # If the top Connections post object has Connections, add those Connections to the stack so we can
                    # traverse it                                                      
                    if top_conn.post_obj in adjacency_list:                            
                        for new_conn in adjacency_list[top_conn.post_obj]:             
                            conn_stack.push(new_conn)                                  

    if num_ensembles_with_no_output_conn > 0:                                          
        print "Warning: Number of ensembles with no output connection == %i\n" % num_ensembles_with_no_output_conn

    # Populate the num_synapses dict                                                   
    for ens, ens_list in ens_ens_conn.items():                                         
        num_neurons_in_connected_ens = sum([e.n_neurons for e in ens_list])            
        num_synapses_per_ens[ens] = ens.n_neurons * num_neurons_in_connected_ens       

    # Total number of neurons in preprocessed model taken from the information returned from executing run_spaun.py
    total_num_neurons = 0                                                           
    for ens in model.all_ensembles:                                       
        total_num_neurons += ens.n_neurons                                          

    total_ensembles=len(model.all_ensembles)

    # Total fanout                                                                  
    total_ens_per_ensemble = sum([len(v) for k,v in ens_ens_conn.items()])          

    # Total number of synapses in preprocessed model                                
    total_num_synapses = sum([v for k,v in num_synapses_per_ens.items()])           

    # Note that mem_dec, tat_rn and tat_enc are values returned from the preprocessing function in nengo_brainstorm_pp module.
    encoding_weights, decoding_weights, transform_count = calc_new_costs(model)
    
    # Total decoders                                                                
    total_decoding_weights = decoding_weights

    # Total encoders                                                                
    total_encoding_weights = encoding_weights
    
    # Total transform resource count
    total_transform_resource_count = transform_count

    total_mem_count = total_transform_resource_count + total_encoding_weights + total_decoding_weights
    compression = float(total_mem_count) / float(total_num_synapses)

    # average number of neurons per ensemble                                        
    avg_nrn_per_ensemble = np.mean([e.n_neurons for e in model.all_ensembles])

    # average fanout                                                                
    avg_ens_per_ensemble = np.mean([len(v) for k,v in ens_ens_conn.items()])        

    # average number of synapses per neuron                                         
    avg_syn_per_neuron = total_num_synapses / float(total_num_neurons)                     

    if print_results:
        print "Total number of neurons = %i" % total_num_neurons                        
        print "Total number of ensembles = %i" % total_ensembles
        print "Total number of synapses = %i" % total_num_synapses                      
        print "Total fanout = %i" % total_ens_per_ensemble                              
        print "Total decoding weights = %i" % total_decoding_weights                    
        print "Total encoding weights = %i" % total_encoding_weights
        print "Total transform resource count = %i\n" % total_transform_resource_count
        print "Compression = %0.6f\n" % compression
        print "Average number of neurons per ensemble = %.2f" % avg_nrn_per_ensemble    
        print "Average number of synapses per neuron = %.2f" % avg_syn_per_neuron       
        print "Average fanout = %.2f" % avg_ens_per_ensemble

    return { "N_NRN":total_num_neurons, "N_ENS": total_ensembles, \
            "N_SYN": total_num_synapses, "N_FAN": total_ens_per_ensemble, \
            "N_D": total_decoding_weights, "N_E": total_encoding_weights, \
            "N_T": total_transform_resource_count, "C": compression, \
            "R_NRN_ENS": avg_nrn_per_ensemble, "R_SYN_NRN": avg_syn_per_neuron,\
            "R_FAN_ENS": avg_ens_per_ensemble }
