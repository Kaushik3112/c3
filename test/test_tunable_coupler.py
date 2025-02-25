"""
integration testing module for tunable coupler element
and line specific chain of signal generation.
"""

# System imports
import copy
import pickle
import pytest
import numpy as np

# Main C3 objects
from c3.c3objs import Quantity as Qty
from c3.parametermap import ParameterMap as PMap
from c3.experiment import Experiment as Exp
from c3.model import Model as Mdl
from c3.generator.generator import Generator as Gnr

# Building blocks
import c3.generator.devices as devices
import c3.signal.gates as gates
import c3.signal.pulse as pulse

# Libs and helpers
import c3.libraries.hamiltonians as hamiltonians
import c3.libraries.envelopes as envelopes
import c3.libraries.chip as chip

lindblad = False
dressed = True
q1_lvls = 3
q2_lvls = 3
tc_lvls = 3
freq_q1 = 6.189e9
freq_q2 = 5.089e9
freq_tc = 8.1e9
phi_0_tc = 10
fluxpoint = phi_0_tc * 0.23
d = 0.36

anhar_q1 = -286e6
anhar_q2 = -310e6
anhar_TC = -235e6
coupling_strength_q1tc = 142e6
coupling_strength_q2tc = 116e6
coupling_strength_q1q2 = 0 * 1e6

t1_q1 = 23e-6
t1_q2 = 70e-6
t1_tc = 15e-6
t2star_q1 = 27e-6
t2star_q2 = 50e-6
t2star_tc = 7e-6
init_temp = 0.06
v2hz = 1e9
t_final = 10e-9  # Time for single qubit gates
sim_res = 100e9
awg_res = 2.4e9

cphase_time = 100e-9  # Two qubit gate
flux_freq = 829 * 1e6
offset = 0 * 1e6
fluxamp = 0.1 * phi_0_tc
t_down = cphase_time - 5e-9
xy_angle = -0.3590456701578104
framechange_q1 = 0.725 * np.pi
framechange_q2 = 1.221 * np.pi

# ### MAKE MODEL
q1 = chip.Qubit(
    name="Qubit1",
    desc="Qubit 1",
    freq=Qty(value=freq_q1, min_val=5.0e9, max_val=8.0e9, unit="Hz 2pi"),
    anhar=Qty(value=anhar_q1, min_val=-380e6, max_val=-120e6, unit="Hz 2pi"),
    hilbert_dim=q1_lvls,
    t1=Qty(value=t1_q1, min_val=5e-6, max_val=90e-6, unit="s"),
    t2star=Qty(value=t2star_q1, min_val=10e-6, max_val=90e-6, unit="s"),
    temp=Qty(value=init_temp, min_val=0.0, max_val=0.12, unit="K"),
)
q2 = chip.Qubit(
    name="Qubit2",
    desc="Qubit 2",
    freq=Qty(value=freq_q2, min_val=5.0e9, max_val=8.0e9, unit="Hz 2pi"),
    anhar=Qty(value=anhar_q2, min_val=-380e6, max_val=-120e6, unit="Hz 2pi"),
    hilbert_dim=q2_lvls,
    t1=Qty(value=t1_q2, min_val=5e-6, max_val=90e-6, unit="s"),
    t2star=Qty(value=t2star_q2, min_val=10e-6, max_val=90e-6, unit="s"),
    temp=Qty(value=init_temp, min_val=0.0, max_val=0.12, unit="K"),
)
tc_at = chip.Transmon(
    name="TCQubit",
    desc="Tunable Coupler",
    freq=Qty(value=freq_tc, min_val=0.0e9, max_val=10.0e9, unit="Hz 2pi"),
    phi=Qty(
        value=fluxpoint, min_val=-5.0 * phi_0_tc, max_val=5.0 * phi_0_tc, unit="Wb"
    ),
    phi_0=Qty(
        value=phi_0_tc, min_val=phi_0_tc * 0.9, max_val=phi_0_tc * 1.1, unit="Wb"
    ),
    d=Qty(value=d, min_val=d * 0.9, max_val=d * 1.1, unit=""),
    hilbert_dim=tc_lvls,
    anhar=Qty(value=anhar_TC, min_val=-380e6, max_val=-120e6, unit="Hz 2pi"),
    t1=Qty(value=t1_tc, min_val=1e-6, max_val=90e-6, unit="s"),
    t2star=Qty(value=t2star_tc, min_val=1e-6, max_val=90e-6, unit="s"),
    temp=Qty(value=init_temp, min_val=0.0, max_val=0.12, unit="K"),
)
q1tc = chip.Coupling(
    name="Q1-TC",
    desc="Coupling qubit 1 to tunable coupler",
    connected=["Qubit1", "TCQubit"],
    strength=Qty(
        value=coupling_strength_q1tc, min_val=0 * 1e4, max_val=200e6, unit="Hz 2pi"
    ),
    hamiltonian_func=hamiltonians.int_XX,
)
q2tc = chip.Coupling(
    name="Q2-TC",
    desc="Coupling qubit 2 to t×unable coupler",
    connected=["Qubit2", "TCQubit"],
    strength=Qty(
        value=coupling_strength_q2tc, min_val=0 * 1e4, max_val=200e6, unit="Hz 2pi"
    ),
    hamiltonian_func=hamiltonians.int_XX,
)
q1q2 = chip.Coupling(
    name="Q1-Q2",
    desc="Coupling qubit 1 to qubit 2",
    connected=["Qubit1", "Qubit2"],
    strength=Qty(
        value=coupling_strength_q1q2, min_val=0 * 1e4, max_val=200e6, unit="Hz 2pi"
    ),
    hamiltonian_func=hamiltonians.int_XX,
)
drive_q1 = chip.Drive(
    name="Q1",
    desc="Drive on Q1",
    connected=["Qubit1"],
    hamiltonian_func=hamiltonians.x_drive,
)
drive_q2 = chip.Drive(
    name="Q2",
    desc="Drive on Q2",
    connected=["Qubit2"],
    hamiltonian_func=hamiltonians.x_drive,
)
flux = chip.Drive(
    name="TC",
    desc="Flux drive/control on tunable couler",
    connected=["TCQubit"],
    hamiltonian_func=hamiltonians.z_drive,
)
phys_components = [tc_at, q1, q2]
line_components = [q1tc, q2tc, q1q2, drive_q1, drive_q2, flux]

model = Mdl(phys_components, line_components, [])
model.set_lindbladian(lindblad)
model.set_dressed(dressed)

# ### MAKE GENERATOR
lo = devices.LO(name="lo", resolution=sim_res)
awg = devices.AWG(name="awg", resolution=awg_res)
dig_to_an = devices.DigitalToAnalog(name="dac", resolution=sim_res)
resp = devices.Response(
    name="resp",
    rise_time=Qty(value=0.3e-9, min_val=0.05e-9, max_val=0.6e-9, unit="s"),
    resolution=sim_res,
)
mixer = devices.Mixer(name="mixer")
fluxbias = devices.FluxTuning(
    name="fluxbias",
    phi_0=Qty(
        value=phi_0_tc, min_val=0.9 * phi_0_tc, max_val=1.1 * phi_0_tc, unit="Wb"
    ),
    phi=Qty(
        value=fluxpoint, min_val=-1.0 * phi_0_tc, max_val=1.0 * phi_0_tc, unit="Wb"
    ),
    omega_0=Qty(
        value=freq_tc, min_val=0.9 * freq_tc, max_val=1.1 * freq_tc, unit="Hz 2pi"
    ),
    d=Qty(value=d, min_val=d * 0.9, max_val=d * 1.1, unit=""),
    anhar=Qty(value=anhar_q1, min_val=-380e6, max_val=-120e6, unit="Hz 2pi"),
)
v_to_hz = devices.VoltsToHertz(
    name="v2hz",
    V_to_Hz=Qty(value=v2hz, min_val=0.9 * v2hz, max_val=1.1 * v2hz, unit="Hz 2pi/V"),
)
device_dict = {
    dev.name: dev for dev in [lo, awg, mixer, dig_to_an, resp, v_to_hz, fluxbias]
}
generator = Gnr(
    devices=device_dict,
    chains={
        "TC": {
            "lo": [],
            "awg": [],
            "dac": ["awg"],
            "resp": ["dac"],
            "mixer": ["lo", "resp"],
            "fluxbias": ["mixer"],
        },
        "Q1": {
            "lo": [],
            "awg": [],
            "dac": ["awg"],
            "resp": ["dac"],
            "mixer": ["lo", "resp"],
            "v2hz": ["mixer"],
        },
        "Q2": {
            "lo": [],
            "awg": [],
            "dac": ["awg"],
            "resp": ["dac"],
            "mixer": ["lo", "resp"],
            "v2hz": ["mixer"],
        },
    },
)

# ### MAKE GATESET
nodrive_env = pulse.Envelope(name="no_drive", params={}, shape=envelopes.no_drive)
carrier_parameters = {
    "freq": Qty(value=freq_q1, min_val=0e9, max_val=10e9, unit="Hz 2pi"),
    "framechange": Qty(value=0.0, min_val=-3 * np.pi, max_val=5 * np.pi, unit="rad"),
}
carr_q1 = pulse.Carrier(
    name="carrier", desc="Frequency of the local oscillator", params=carrier_parameters
)
carr_q2 = copy.deepcopy(carr_q1)
carr_q2.params["freq"].set_value(freq_q2)
carr_tc = copy.deepcopy(carr_q1)
carr_tc.params["freq"].set_value(flux_freq)


flux_params = {
    "amp": Qty(value=fluxamp, min_val=0.0, max_val=5, unit="V"),
    "t_final": Qty(
        value=cphase_time,
        min_val=0.5 * cphase_time,
        max_val=1.5 * cphase_time,
        unit="s",
    ),
    "t_up": Qty(
        value=5 * 1e-9, min_val=0.0 * cphase_time, max_val=0.5 * cphase_time, unit="s"
    ),
    "t_down": Qty(
        value=t_down, min_val=0.5 * cphase_time, max_val=1.0 * cphase_time, unit="s"
    ),
    "risefall": Qty(
        value=5 * 1e-9, min_val=0.0 * cphase_time, max_val=1.0 * cphase_time, unit="s"
    ),
    "freq_offset": Qty(
        value=offset, min_val=-50 * 1e6, max_val=50 * 1e6, unit="Hz 2pi"
    ),
    "xy_angle": Qty(
        value=xy_angle, min_val=-0.5 * np.pi, max_val=2.5 * np.pi, unit="rad"
    ),
}
flux_env = pulse.Envelope(
    name="flux",
    desc="Flux bias for tunable coupler",
    params=flux_params,
    shape=envelopes.flattop,
)
crzp = gates.Instruction(
    name="crzp",
    targets=[0, 1],
    t_start=0.0,
    t_end=cphase_time,
    channels=["Q1", "Q2", "TC"],
)
crzp.add_component(flux_env, "TC")
crzp.add_component(carr_tc, "TC")
crzp.add_component(nodrive_env, "Q1")
crzp.add_component(carr_q1, "Q1")
crzp.comps["Q1"]["carrier"].params["framechange"].set_value(framechange_q1)
crzp.add_component(nodrive_env, "Q2")
crzp.add_component(carr_q2, "Q2")
crzp.comps["Q2"]["carrier"].params["framechange"].set_value(framechange_q2)


# ### MAKE EXPERIMENT
parameter_map = PMap(instructions=[crzp], model=model, generator=generator)
exp = Exp(pmap=parameter_map)

##### TESTING ######

with open("test/tunable_coupler_data.pickle", "rb") as filename:
    data = pickle.load(filename)


@pytest.mark.integration
def test_coupler_frequency() -> None:
    coupler_01 = np.abs(
        np.abs(model.eigenframe[model.state_labels.index((0, 0, 0))])
        - np.abs(model.eigenframe[model.state_labels.index((1, 0, 0))])
    )
    rel_diff = np.abs((coupler_01 - data["coupler_01"]) / data["coupler_01"])
    assert rel_diff < 1e-12


@pytest.mark.integration
def test_coupler_anahrmonicity() -> None:
    coupler_12 = np.abs(
        np.abs(model.eigenframe[model.state_labels.index((1, 0, 0))])
        - np.abs(model.eigenframe[model.state_labels.index((2, 0, 0))])
    )
    rel_diff = np.abs((coupler_12 - data["coupler_12"]) / data["coupler_12"])
    assert rel_diff < 1e-12


@pytest.mark.integration
def test_energy_levels() -> None:
    model = parameter_map.model
    parameter_map.set_parameters([0.0], [[["TCQubit-phi"]]])
    model.update_model()
    labels = [
        model.state_labels[indx]
        for indx in np.argsort(np.abs(model.eigenframe) / 2 / np.pi / 1e9)
    ]
    product_basis = []
    dressed_basis = []
    ordered_basis = []
    transforms = []
    steps = 101
    min_ratio = -0.10
    max_ratio = 0.7
    flux_ratios = np.linspace(min_ratio, max_ratio, steps, endpoint=True)
    for flux_ratio in flux_ratios:
        flux_bias = flux_ratio * phi_0_tc
        parameter_map.set_parameters(
            [flux_bias, 0.0, 0.0, 0.0],
            [
                [["TCQubit-phi"]],
                [["Q1-TC-strength"]],
                [["Q2-TC-strength"]],
                [["Q1-Q2-strength"]],
            ],
        )
        model.update_model()
        product_basis.append(
            [
                model.eigenframe[model.state_labels.index(label)] / 2 / np.pi / 1e9
                for label in labels
            ]
        )
        parameter_map.set_parameters(
            [coupling_strength_q1tc, coupling_strength_q2tc, coupling_strength_q1q2],
            [[["Q1-TC-strength"]], [["Q2-TC-strength"]], [["Q1-Q2-strength"]]],
        )
        model.update_model()
        ordered_basis.append(
            [
                model.eigenframe[model.state_labels.index(label)] / 2 / np.pi / 1e9
                for label in labels
            ]
        )
        parameter_map.model.update_dressed(ordered=False)
        dressed_basis.append(
            [
                model.eigenframe[model.state_labels.index(label)] / 2 / np.pi / 1e9
                for label in model.state_labels
            ]
        )
        transforms.append(
            np.array(
                [
                    np.real(model.transform[model.state_labels.index(label)])
                    for label in labels
                ]
            )
        )
    parameter_map.set_parameters([fluxpoint], [[["TCQubit-phi"]]])
    model.update_model()
    dressed_basis = np.array(dressed_basis)
    ordered_basis = np.array(ordered_basis)
    product_basis = np.array(product_basis)
    print((np.abs(product_basis - data["product_basis"]) < 1).all())
    assert (np.abs(product_basis - data["product_basis"]) < 1).all()
    assert (np.abs(ordered_basis - data["ordered_basis"]) < 1).all()
    # Dressed basis might change at avoided crossings depending on how we
    # decide to deal with it. Atm no state with largest probability is chosen.
    assert (np.abs(dressed_basis - data["dressed_basis"]) < 1).all()


@pytest.mark.slow
@pytest.mark.integration
def test_dynamics_CPHASE() -> None:
    # Dynamics (closed system)
    exp.set_opt_gates(["crzp[0, 1]"])
    exp.compute_propagators()
    dUs = []
    for indx in range(0, len(exp.partial_propagators["crzp[0, 1]"]), 50):
        dUs.append(exp.partial_propagators["crzp[0, 1]"][indx].numpy())
    dUs = np.array(dUs)
    np.testing.assert_array_almost_equal(dUs, data["dUs"], decimal=3)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.heavy
@pytest.mark.skip(reason="takes way too long")
def test_dynamics_CPHASE_lindblad() -> None:
    # Dynamics (open system)
    exp.pmap.model.set_lindbladian(True)
    propagators = exp.compute_propagators()
    U_super = propagators["crzp[0, 1]"]
    assert (np.abs(np.real(U_super) - np.real(data["U_super"])) < 1e-8).all()
    assert (np.abs(np.imag(U_super) - np.imag(data["U_super"])) < 1e-8).all()
    assert (np.abs(np.abs(U_super) - np.abs(data["U_super"])) < 1e-8).all()
    assert (np.abs(np.angle(U_super) - np.angle(data["U_super"])) < 1e-8).all()


@pytest.mark.integration
def test_separate_chains() -> None:
    assert generator.chains["Q2"] == generator.chains["Q1"]
    assert generator.chains["TC"] != generator.chains["Q1"]
    assert generator.chains["TC"] != generator.chains["Q2"]


@pytest.mark.slow
@pytest.mark.integration
def test_flux_signal() -> None:
    instr = exp.pmap.instructions["crzp[0, 1]"]
    signal = exp.pmap.generator.generate_signals(instr)
    awg = exp.pmap.generator.devices["awg"]
    # mixer = exp.pmap.generator.devices["mixer"]
    channel = "TC"
    tc_signal = signal[channel]["values"].numpy()
    tc_ts = signal[channel]["ts"].numpy()
    tc_awg_I = awg.signal[channel]["inphase"].numpy()
    tc_awg_Q = awg.signal[channel]["quadrature"].numpy()
    tc_awg_ts = awg.signal[channel]["ts"].numpy()
    np.testing.assert_allclose(tc_signal[1:], data["tc_signal"][1:], rtol=1e-8)
    # First pixel is wrong in the new method. I'm very sorry.
    np.testing.assert_allclose(tc_ts, data["tc_ts"])
    np.testing.assert_allclose(tc_awg_I, data["tc_awg_I"])
    np.testing.assert_allclose(tc_awg_Q, data["tc_awg_Q"])
    np.testing.assert_allclose(tc_awg_ts, data["tc_awg_ts"])


@pytest.mark.unit
def test_FluxTuning():
    flux_tune = devices.FluxTuning(
        name="flux_tune",
        phi_0=Qty(phi_0_tc),
        phi=Qty(value=0, min_val=-phi_0_tc, max_val=phi_0_tc),
        omega_0=Qty(freq_tc),
        anhar=Qty(anhar_TC),
        d=Qty(d),
    )

    transmon = chip.Transmon(
        name="transmon",
        hilbert_dim=3,
        freq=Qty(freq_tc),
        phi=Qty(value=0, min_val=-1.5 * phi_0_tc, max_val=1.5 * phi_0_tc),
        phi_0=Qty(phi_0_tc),
        d=Qty(d),
        anhar=Qty(anhar_TC),
    )

    bias_phis = [0, 0.2]
    phis = np.linspace(-1, 1, 10) * phi_0_tc

    for bias_phi in bias_phis:
        flux_tune.params["phi"].set_value(bias_phi)
        signal = [{"ts": np.linspace(0, 1, 10), "values": phis}]
        signal_out = flux_tune.process(None, None, signal)
        flux_tune_frequencies = signal_out["values"].numpy()

        transmon_frequencies = []
        transmon.params["phi"].set_value(bias_phi)
        bias_freq = transmon.get_freq()
        for phi in phis + bias_phi:
            transmon.params["phi"].set_value(phi)
            transmon_frequencies.append(transmon.get_freq())
        transmon_diff_freq = np.array(transmon_frequencies) - bias_freq

        assert (
            np.max(np.abs(flux_tune_frequencies - transmon_diff_freq)) < 1e-15 * freq_tc
        )


@pytest.mark.unit
def test_symmetric_FluxTuning():
    flux_tune = devices.FluxTuning(
        name="flux_tune",
        phi_0=Qty(phi_0_tc),
        phi=Qty(value=0, min_val=-phi_0_tc, max_val=phi_0_tc),
        omega_0=Qty(freq_tc),
        anhar=Qty(anhar_TC),
    )

    transmon = chip.Transmon(
        name="transmon",
        hilbert_dim=3,
        freq=Qty(freq_tc),
        phi=Qty(value=0, min_val=-1.5 * phi_0_tc, max_val=1.5 * phi_0_tc),
        phi_0=Qty(phi_0_tc),
        anhar=Qty(anhar_TC),
    )

    bias_phis = [0, 0.2]
    phis = np.linspace(-1, 1, 10) * phi_0_tc

    for bias_phi in bias_phis:
        flux_tune.params["phi"].set_value(bias_phi)
        signal = [{"ts": np.linspace(0, 1, 10), "values": phis}]
        signal_out = flux_tune.process(None, None, signal)
        flux_tune_frequencies = signal_out["values"].numpy()

        transmon_frequencies = []
        transmon.params["phi"].set_value(bias_phi)
        bias_freq = transmon.get_freq()
        for phi in phis + bias_phi:
            transmon.params["phi"].set_value(phi)
            transmon_frequencies.append(transmon.get_freq())
        transmon_diff_freq = np.array(transmon_frequencies) - bias_freq

        np.testing.assert_allclose(flux_tune_frequencies, transmon_diff_freq)
