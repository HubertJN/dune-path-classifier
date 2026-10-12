import uproot, numpy as np

# Helper classes
class Events:
    def __init__(self, filename):
        file = uproot.open(filename)
        tree = file["EventTree"]
        # event
        self.event_number = tree['eventId'].array(library="np")
        self.interaction_type = tree['interactionType'].array(library="np")
        self.num_final_state_particles = tree['nFinalStateParticles'].array(library="np")
        # reco particle
        self.reco_particle_index = tree['pfoId'].array(library="np")
        self.reco_parent_index = tree['parent'].array(library="np")
        self.reco_children_indices = tree['children'].array(library="np")
        self.reco_num_hits_total = tree['nHitsInPfoTotal'].array(library="np")
        self.reco_num_hits_u = tree['nHitsInPfoU'].array(library="np")
        self.reco_num_hits_v = tree['nHitsInPfoV'].array(library="np")
        self.reco_num_hits_w = tree['nHitsInPfoW'].array(library="np")
        self.reco_hits_x_u = tree['xU'].array(library="np")
        self.reco_hits_x_v = tree['xV'].array(library="np")
        self.reco_hits_x_w = tree['xW'].array(library="np")
        self.reco_hits_u = tree['u'].array(library="np")
        self.reco_hits_v = tree['v'].array(library="np")
        self.reco_hits_w = tree['w'].array(library="np")
        self.reco_adcs_u = tree['adcU'].array(library="np")
        self.reco_adcs_v = tree['adcV'].array(library="np")
        self.reco_adcs_w = tree['adcW'].array(library="np")
        self.reco_hits_3d_x = tree['x3d'].array(library="np")
        self.reco_hits_3d_y = tree['y3d'].array(library="np")
        self.reco_hits_3d_z = tree['z3d'].array(library="np")
        self.reco_particle_vtx_3d_x = tree['vtxX3d'].array(library="np")
        self.reco_particle_vtx_3d_y = tree['vtxY3d'].array(library="np")
        self.reco_particle_vtx_3d_z = tree['vtxZ3d'].array(library="np")
        self.reco_particle_vtx_x = tree['vtxX'].array(library="np")
        self.reco_particle_vtx_u = tree['vtxU'].array(library="np")
        self.reco_particle_vtx_v = tree['vtxV'].array(library="np")
        self.reco_particle_vtx_w = tree['vtxW'].array(library="np")
        # mc particle
        self.mc_num_hits_total = tree['nHitsInBestMCParticleTotal'].array(library="np")
        self.mc_num_hits_u = tree['nHitsInBestMCParticleU'].array(library="np")
        self.mc_num_hits_v = tree['nHitsInBestMCParticleV'].array(library="np")
        self.mc_num_hits_w = tree['nHitsInBestMCParticleW'].array(library="np")
        self.mc_pdg = tree['bestMCParticlePdgCode'].array(library="np")
        # metrics
        self.purity = tree['purity'].array(library="np")
        self.completeness = tree['completeness'].array(library="np")
        # neutrino
        self.neutrino_vtx_3d_x = tree['nuVtxX3d'].array(library="np")
        self.neutrino_vtx_3d_y = tree['nuVtxY3d'].array(library="np")
        self.neutrino_vtx_3d_z = tree['nuVtxZ3d'].array(library="np")
        self.neutrino_vtx_x = tree['nuVtxX'].array(library="np")
        self.neutrino_vtx_u = tree['nuVtxU'].array(library="np")
        self.neutrino_vtx_v = tree['nuVtxV'].array(library="np")
        self.neutrino_vtx_w = tree['nuVtxW'].array(library="np")
        self.true_neutrino_vtx_3d_x = tree['trueNuVtxX3d'].array(library="np")
        self.true_neutrino_vtx_3d_y = tree['trueNuVtxY3d'].array(library="np")
        self.true_neutrino_vtx_3d_z = tree['trueNuVtxZ3d'].array(library="np")
        self.true_neutrino_vtx_x = tree['trueNuVtxX'].array(library="np")
        self.true_neutrino_vtx_u = tree['trueNuVtxU'].array(library="np")
        self.true_neutrino_vtx_v = tree['trueNuVtxV'].array(library="np")
        self.true_neutrino_vtx_w = tree['trueNuVtxW'].array(library="np")
        file.close()
        self.make_sequential()

    def filter_by_event(self, input_var, event_list):
        if type(event_list) == int:
            return input_var[np.where(self.event_number == event_list)]
        elif type(event_list) == list:
            if not np.all([ isinstance(val, int) for val in event_list ]):
                raise TypeError(f'event_list must be int or list of ints, received {type(event_list)}: {event_list}')
            mask = np.zeros_like(input_var)
            for event_number in event_list:
                mask |= (self.event_number == event_number)
            return input_var[np.where(mask)]
        else:
            raise TypeError(f'event_list must be an int or list of ints, received {type(event_list)}: {event_list}')

    def filter_by_pdg(self, input_var, pdg_list, respect_sign=False):
        if type(pdg_list) == int:
            mc_pdg = np.abs(self.mc_pdg) if not respect_sign else self.mc_pdg
            return input_var[np.where(mc_pdg == pdg_list)]
        elif type(pdg_list) == list:
            if not np.all([ isinstance(val, int) for val in pdg_list ]):
                raise TypeError(f'pdg_list must be int or list of ints, received {type(pdg_list)}: {pdg_list}')
            mc_pdg = np.abs(self.mc_pdg) if not respect_sign else self.mc_pdg
            mask = np.zeros_like(input_var)
            for pdg in pdg_list:
                mask |= (mc_pdg == pdg)
            return input_var[np.where(mask)]
        else:
            raise TypeError(f'pdg_list must be an int or list of ints, received {type(pdg_list)}: {pdg_list}')

    def make_sequential(self):
        seq_events = np.zeros_like(self.event_number)
        seq_events[0] = self.event_number[0]
        seq_event = 0
        for i in range(1, len(self.event_number)):
            if self.event_number[i] != self.event_number[i - 1]:
                seq_event += 1
            seq_events[i] = seq_event
        self.event_number = seq_events


from awkward import Array

class View:
    def __init__(self, events, view):
        if type(events) != Events:
            raise Exception("Parameter 'events' not of type Events")
        if view.lower() not in ["u", "v", "w"]:
            raise Exception("Parameter 'view' not one of u, v or w")
        all_events = np.unique(events.event_number)
        self.true_vtx_x = np.array([ events.true_neutrino_vtx_x[np.where(events.event_number == e)][0]
                                           for e in all_events ])
        if view.lower() == "u":
            self.x = [ np.concatenate(events.reco_hits_x_u[np.where(events.event_number == e)])
                                           for e in all_events ]
            self.z = [ np.concatenate(events.reco_hits_u[np.where(events.event_number == e)])
                                           for e in all_events ]
            self.adc = [ np.concatenate(events.reco_adcs_u[np.where(events.event_number == e)])
                                             for e in all_events ]
            self.true_vtx_z = np.array([ events.neutrino_vtx_u[np.where(events.event_number == e)][0]
                                        for e in all_events ])
        elif view.lower() == "v":
            self.x = [ np.concatenate(events.reco_hits_x_v[np.where(events.event_number == e)])
                                           for e in all_events ]
            self.z = [ np.concatenate(events.reco_hits_v[np.where(events.event_number == e)])
                                           for e in all_events ]
            self.adc = [ np.concatenate(events.reco_adcs_v[np.where(events.event_number == e)])
                                             for e in all_events ]
            self.true_vtx_z = np.array([ events.neutrino_vtx_v[np.where(events.event_number == e)][0]
                                        for e in all_events ])
        else:
            self.x = [ np.concatenate(events.reco_hits_x_w[np.where(events.event_number == e)])
                                           for e in all_events ]
            self.z = [ np.concatenate(events.reco_hits_w[np.where(events.event_number == e)])
                                           for e in all_events ]
            self.adc = [ np.concatenate(events.reco_adcs_w[np.where(events.event_number == e)])
                                             for e in all_events ]
            self.true_vtx_z = np.array([ events.true_neutrino_vtx_w[np.where(events.event_number == e)][0]
                                        for e in all_events ])