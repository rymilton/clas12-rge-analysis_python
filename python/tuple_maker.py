import uproot
import awkward as ak
import numpy as np
import argparse
import os 

beam_energy = 10.5473 # Number taken from RCDB. Doesn't seem to be in hipo files

output_branch_names = {
    "REC::Particle::beta" : "beta",
    "REC::Particle::charge" : "charge",
    "REC::Particle::chi2pid" : "chi2pid",
    "REC::Particle::pid" : "pid",
    "REC::Particle::px" : "p_x",
    "REC::Particle::py" : "p_y",
    "REC::Particle::pz" : "p_z",
    "total_particle_momentum" : "p",
    "particle_mass" : "mass",
    "REC::Particle::status" : "status",
    "REC::Particle::vt" : "v_t",
    "REC::Particle::vx" : "v_x",
    "REC::Particle::vy" : "v_y",
    "REC::Particle::vz" : "v_z",
    "particle_theta" : "theta",
    "particle_phi" : "phi",
    "REC::Track::sector" : "sector",
    "REC::Track::q" : "track_charge",
    "REC::Track::chi2" : "chi2",
    "REC::Track::ndf" : "NDF",
    "PCAL_energy" : "E_PCAL",
    "ECIN_energy" : "E_ECIN",
    "ECOUT_energy" : "E_ECOU",
    "HTCC_photoelectrons" : "Nphe_HTCC",
    "LTCC_photoelectrons" : "Nphe_LTCC",
    "time_of_flight" : "TOF",
    "beam_energy" : "E_beam",
    "event_number" : "event_num",
    "run_number" : "run_num",
}

def add_to_positions(targets, indices, values, tracker=None):
    result = list(targets)
    if tracker is not None:
        tracker_copy = list(tracker)
    for index, value in zip(indices, values):
        if tracker is not None:
            if not tracker_copy[index]:
                result[index] += value
                tracker_copy[index] = True
        else:
            result[index] += value
    if tracker is not None:
        return result, tracker_copy
    else:
        return result

def get_calorimeter_energy(calorimeter_dict, track_indices):

    calorimeter_energies_dict = {
        output_branch_names[f"{ecal_name}_energy"] : ak.zeros_like(track_indices) for ecal_name in ["PCAL", "ECIN", "ECOUT"]
    }

    ecal_masks = {
        "PCAL" : calorimeter_dict["REC::Calorimeter::layer"] == 1,
        "ECIN" : calorimeter_dict["REC::Calorimeter::layer"] == 4,
        "ECOUT" : calorimeter_dict["REC::Calorimeter::layer"] == 7
    }

    for ecal_name in ["PCAL", "ECIN", "ECOUT"]:
        calorimeter_energies_dict[output_branch_names[f"{ecal_name}_energy"]] = ak.Array([
            add_to_positions(particle_edeps, particle_indices, calorimeter_energies)
            for particle_edeps, particle_indices, calorimeter_energies in zip(
                calorimeter_energies_dict[output_branch_names[f"{ecal_name}_energy"]],
                ak.values_astype(calorimeter_dict["REC::Calorimeter::pindex"][ecal_masks[ecal_name]], np.int32),
                calorimeter_dict["REC::Calorimeter::energy"][ecal_masks[ecal_name]]
            )
        ]) 
    return calorimeter_energies_dict

def get_TOF(TOF_dict, calorimeter_dict, track_indices):
    output_TOF_dict = {output_branch_names["time_of_flight"] : ak.zeros_like(track_indices)}

    FTOF_mask = TOF_dict["REC::Scintillator::detector"] == 12
    combined_calo_tof_layers = ak.concatenate(
        (TOF_dict["REC::Scintillator::layer"][FTOF_mask], 10+calorimeter_dict["REC::Calorimeter::layer"]),
        axis = 1,
    )
    combined_calo_tof_indices = ak.concatenate(
        (TOF_dict["REC::Scintillator::pindex"][FTOF_mask], calorimeter_dict["REC::Calorimeter::pindex"]),
        axis = 1,
    )
    combined_calo_tof_times = ak.concatenate(
        (TOF_dict["REC::Scintillator::time"][FTOF_mask], calorimeter_dict["REC::Calorimeter::time"]),
        axis = 1,
    )

    layer_masks = {
        "FTOF1B" : combined_calo_tof_layers == 2,
        "FTOF1A" : combined_calo_tof_layers == 1,
        "FTOF2" : combined_calo_tof_layers == 3,
        "PCAL" : combined_calo_tof_layers == 11,
        "ECIN" : combined_calo_tof_layers == 14,
        "ECOUT" : combined_calo_tof_layers == 17,
    }

    particle_tof_tracker = ak.full_like(track_indices, False, dtype=bool)

    for layer_names in ["FTOF1B", "FTOF1A", "FTOF2", "PCAL", "ECIN", "ECOUT"]:
        time_of_flights_result = [
            add_to_positions(particle_tofs, particle_indices, detector_tofs, tracker=tracker)
            for particle_tofs, tracker, particle_indices, detector_tofs in zip(
                output_TOF_dict[output_branch_names["time_of_flight"]],
                particle_tof_tracker,
                ak.values_astype(combined_calo_tof_indices[layer_masks[layer_names]], np.int32),
                combined_calo_tof_times[layer_masks[layer_names]],
            )
        ]
        output_TOF_dict[output_branch_names["time_of_flight"]] = [item[0] for item in time_of_flights_result]
        particle_tof_tracker = [item[1] for item in time_of_flights_result]


    return output_TOF_dict
def get_cherenkov_counts(cherenkov_dict, track_indices):

    cherenkov_counts_dict = {
        output_branch_names[f"{counter_name}_photoelectrons"] : ak.zeros_like(track_indices) for counter_name in ["HTCC", "LTCC"]
    }

    cherenkov_masks = {
        "HTCC" : cherenkov_dict["REC::Cherenkov::detector"] == 15,
        "LTCC" : cherenkov_dict["REC::Cherenkov::detector"] == 16,
    }

    for counter_name in ["HTCC", "LTCC"]:
        cherenkov_counts_dict[output_branch_names[f"{counter_name}_photoelectrons"]] = ak.Array([
            add_to_positions(particle_photoelectrons, particle_indices, cherenkov_photoelectrons)
            for particle_photoelectrons, particle_indices, cherenkov_photoelectrons in zip(
                cherenkov_counts_dict[output_branch_names[f"{counter_name}_photoelectrons"]],
                ak.values_astype(cherenkov_dict["REC::Cherenkov::pindex"][cherenkov_masks[counter_name]], np.int32),
                cherenkov_dict["REC::Cherenkov::nphe"][cherenkov_masks[counter_name]]
            )
        ])
    return cherenkov_counts_dict

def calculate_momentum(px, py, pz):
    return np.sqrt(px**2 + py**2 + pz**2)

def calculate_theta_lab(px, py, pz):
    return np.arctan2(np.sqrt(px**2 + py**2), pz)

def calculate_phi_lab(px, py):
    return np.arctan2(py, px)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_folder",
        default="/work/clas12/rmilton/clas12-rge-analysis_python",
        help="Folder containing ROOT files with HIPO banks",
    )
    parser.add_argument(
        "--input_file",
        default="banks_00000.root",
        help="Input ROOT file name",
    )
    parser.add_argument(
        "--output_file",
        default="ntuple_output.root",
        help="Output ROOT file name",
    )
    parser.add_argument(
        "--output_folder",
        default="/work/clas12/rmilton/clas12-rge-analysis_python",
        help="Directory to store output files",
    )

    flags = parser.parse_args()

    file_path = os.path.join(flags.data_folder, flags.input_file)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    with uproot.open(f"{file_path}:data") as data:
        print(f"Opening file {file_path}")
        banks = data.arrays()


    # Getting the track indices to match banks to tracks
    track_indices = ak.values_astype(banks["REC::Track::pindex"], np.int32)
    nonempty_track_mask = ak.num(track_indices)>0 # Removing events with no tracks
    track_indices = track_indices[nonempty_track_mask]
    banks = banks[nonempty_track_mask]
    
    tracker_banks_to_readout = ["REC::Track::sector", "REC::Track::q", "REC::Track::ndf", "REC::Track::chi2"]
    tracker_banks_dict = {
        output_branch_names[bank] : banks[bank][track_indices] for bank in tracker_banks_to_readout
    }

    # Getting the particle banks, keeping only particles with a track
    particle_banks_dict = {
        output_branch_names[bank] : banks[bank][track_indices] for bank in banks.fields if "REC::Particle" in bank
    }
    # Calculating other particle quantities
    particle_banks_dict[output_branch_names["total_particle_momentum"]] = calculate_momentum(
        particle_banks_dict[output_branch_names["REC::Particle::px"]],
        particle_banks_dict[output_branch_names["REC::Particle::py"]],
        particle_banks_dict[output_branch_names["REC::Particle::pz"]]
    )

    particle_banks_dict[output_branch_names["particle_theta"]] = calculate_theta_lab(
        particle_banks_dict[output_branch_names["REC::Particle::px"]],
        particle_banks_dict[output_branch_names["REC::Particle::py"]],
        particle_banks_dict[output_branch_names["REC::Particle::pz"]]
    )

    particle_banks_dict[output_branch_names["particle_phi"]] = calculate_phi_lab(
        particle_banks_dict[output_branch_names["REC::Particle::px"]],
        particle_banks_dict[output_branch_names["REC::Particle::py"]],
    )

    # Getting energy deposition for each particle
    calorimeter_indices = ak.values_astype(banks["REC::Calorimeter::pindex"], np.int32)
    calorimeter_track_match = ak.Array([
        [calo_index in track_event for calo_index in calo_event]
        for calo_event, track_event in zip(calorimeter_indices, track_indices)
    ])
    calorimeter_banks_dict = {
        bank : banks[bank][calorimeter_track_match] for bank in banks.fields if "REC::Calorimeter" in bank
    }
    calorimeter_energies_dict = get_calorimeter_energy(calorimeter_banks_dict, track_indices)

    # Getting TOF value for each particle
    TOF_indices = ak.values_astype(banks["REC::Scintillator::pindex"], np.int32)
    TOF_track_match = ak.Array([
        [TOF_index in track_event for TOF_index in TOF_event]
        for TOF_event, track_event in zip(TOF_indices, track_indices)
    ])
    TOF_banks_dict = {
        bank : banks[bank][TOF_track_match] for bank in banks.fields if "REC::Scintillator" in bank
    }
    TOF_dict = get_TOF(TOF_banks_dict, calorimeter_banks_dict, track_indices)
    
    # Getting Cherenkov counter info for each particle
    cherenkov_indices = ak.values_astype(banks["REC::Cherenkov::pindex"], np.int32)
    cherenkov_track_match = ak.Array([
        [cherenkov_index in track_event for cherenkov_index in cherenkov_event]
        for cherenkov_event, track_event in zip(cherenkov_indices, track_indices)
    ])
    cherenkov_banks_dict = {
        bank : banks[bank][cherenkov_track_match] for bank in banks.fields if "REC::Cherenkov" in bank
    }
    cherenkov_photoelectron_dict = get_cherenkov_counts(cherenkov_banks_dict, track_indices)

    # Reading out meta information
    beam_energy_array = np.ones(len(track_indices))*beam_energy
    meta_dict = {output_branch_names["beam_energy"] : beam_energy_array}
    with uproot.recreate(os.path.join(flags.output_folder, flags.output_file)) as file:
        file["data"] = particle_banks_dict | calorimeter_energies_dict | cherenkov_photoelectron_dict | tracker_banks_dict | TOF_dict | meta_dict
    
    '''
        The following banks need the track pindex:
        - Particle banks -- Done
        - Calorimeter banks -- Done
        - Cherenkov banks -- Done
        - TOF banks -- Done
        - 

    '''
    '''
        Needs:
        Associate tracks with particles with the pindex branch
        With the pindex, can get the calorimeter energy, photoelectrons

        Create a JSON file so the user can say what they want to save

        Create DIS variables

        Rename bank variables to match old tuper maker -- Done

        Add MC information


    '''
if __name__ == '__main__':
    main()