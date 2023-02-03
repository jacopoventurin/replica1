import openmm.unit as unit
from openmmtools import cache
import math
import time


class ReplicaStateReporter:
    def __init__(self, file, reportInterval, time=False, potentialEnergy=False, kineticEnergy=False, 
                totalEnergy=False, volume=False, elapsedTime=False, speed=False, separator=',', append=True):
        """
        Create a ReplicaStateReporter.

        Parameters
        ----------
        file : string or file
            The file to write to, specified as a file name or file object
        reportInterval : int
            The interval (in time steps) at which to write frames
        time : bool=False
            Whether to write the current time to the file
        potentialEnergy : bool=False
            Whether to write the potential energy to the file
        kineticEnergy : bool=False
            Whether to write the kinetic energy to the file
        totalEnergy : bool=False
            Whether to write the total energy to the file
        volume : bool=False
            Whether to write the periodic box volume to the file
        elapsedTime : bool=False
            Whether to write the elapsed time of the simulation in seconds to
            the file.
        speed : bool=False
            Whether to write an estimate of the simulation speed in ns/day to
            the file
        separator : string=','
            The separator to use between columns in the file
        append : bool=True
            If true, append to an existing file.  This has two effects.  First,
            the file is opened in append mode.  Second, the header line is not
            written, since there is assumed to already be a header line at the
            start of the file.
        """
        self._reportInterval = reportInterval
        self._openedFile = isinstance(file, str)
        if self._openedFile:
            self._out = open(file, 'a' if append else 'w')
        else:
            self._out = file
       
        self._time = time
        self._potentialEnergy = potentialEnergy
        self._kineticEnergy = kineticEnergy
        self._totalEnergy = totalEnergy
        self._volume = volume
        self._elapsedTime = elapsedTime
        self._speed = speed
        self._separator = separator
        self._append = append
        self._hasInitialized = False
        self._hasTimeInitialized = False
        self._needParameters = volume or time or speed
        self._needEnergy = potentialEnergy or kineticEnergy or totalEnergy

        self.values_list = None


    def store_report(self, thermodynamic_states, sampler_states, temperature_list = None):
        """
        Generate and store in memory a report from thermo_state, sampler_state objects.
        The report can be written in the output file calling report() method.

        Inputs
        ----------
        tehrmodynamic_satate : 
            
        sampler_state : 

        """

        if not self._hasTimeInitialized:
            self._initialSimulationTime = self._getInitialSimulationTime(thermodynamic_states[0], sampler_states[0])#state.getTime()
            self._initialClockTime = time.time()

            self._hasTimeInitialized = True
        
        if self.values_list is None:
            self.values_list = []

        for thermo_state, sampler_state in zip(thermodynamic_states, sampler_states):
            i_t = temperature_list.index(thermo_state.temperature)
            context, _ = cache.global_context_cache.get_context(thermo_state)
            sampler_state.apply_to_context(context)
            state = context.getState(getEnergy=self._needEnergy, getParameters=self._needParameters)

            # Check for errors
            self._checkForErrors(state)

            # Query for the values
            self.values_list.append(self._constructReportValues(state, i_t))



    def report(self):
        """
        Save the report stored in self.value_list to the ouput file.
        If the file was not initialized header is printed.
        """
        if not self._hasInitialized:
            headers = self._constructHeaders()
            #if not self._append:
            print('"%s"' % ('"'+self._separator+'"').join(headers), file=self._out)
            try:
                self._out.flush()
            except AttributeError:
                pass
            
            self._hasInitialized = True

        if self.values_list is not None:
            for values in self.values_list:
                # Write the values.
                print(self._separator.join(str(v) for v in values), file=self._out)
                try:
                    self._out.flush()
                except AttributeError:
                    pass
            del self.values_list
            self.values_list = None
        else:
            pass

    def get_report_interval(self):
        """
        Return the report interval
        """
        return self._reportInterval

    def _getInitialSimulationTime(self, thermo_state, sampler_state):
        """
        Compute initial simulation time from a particular thermo_state and the corresponding sampler state
        """
        context, _ = cache.global_context_cache.get_context(thermo_state)
        sampler_state.apply_to_context(context)
        state = context.getState(getParameters=self._needParameters)

        return state.getTime()

    def _constructReportValues(self, state, index):
        """Query the simulation for the current state of our observables of interest.

        Parameters
        ----------
        state : State
            The current state of the simulation
        index:
            index of the replica

        Returns
        -------
        A list of values summarizing the current state of
        the simulation, to be printed or saved. Each element in the list
        corresponds to one of the columns in the resulting CSV file.
        The first element always represent the index of the replica.
        """
        values = []
        
        # First append the replica index
        values.append(index)
        if self._time:
            values.append(state.getTime().value_in_unit(unit.picosecond))
        if self._potentialEnergy:
            values.append(state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole))
        if self._kineticEnergy:
            values.append(state.getKineticEnergy().value_in_unit(unit.kilojoules_per_mole))
        if self._totalEnergy:
            values.append((state.getKineticEnergy()+state.getPotentialEnergy()).value_in_unit(unit.kilojoules_per_mole))
        if self._volume:
            box = state.getPeriodicBoxVectors()
            volume = box[0][0]*box[1][1]*box[2][2]
            values.append(volume.value_in_unit(unit.nanometer**3))
        if self._elapsedTime:
            values.append(time.time() - self._initialClockTime)
        if self._speed:
            elapsedDays = (time.time()-self._initialClockTime)/86400.0
            elapsedNs = (state.getTime()-self._initialSimulationTime).value_in_unit(unit.nanosecond)
            if elapsedDays > 0.0:
                values.append('%.3g' % (elapsedNs/elapsedDays))
            else:
                values.append('--')

        return values


    def _checkForErrors(self, state):
        """Check for errors in the current state of the simulation

        Parameters
         - simulation (Simulation) The Simulation to generate a report for
         - state (State) The current state of the simulation
        """
        if self._needEnergy:
            energy = (state.getKineticEnergy()+state.getPotentialEnergy()).value_in_unit(unit.kilojoules_per_mole)
            if math.isnan(energy):
                raise ValueError('Energy is NaN')
            if math.isinf(energy):
                raise ValueError('Energy is infinite')

    def _constructHeaders(self):
        """
        Construct the headers for the CSV output

        Returns: a list of strings giving the title of each observable being reported on.
        """
        headers = []
        headers.append('Number Replica')
        if self._time:
            headers.append('Time (ps)')
        if self._potentialEnergy:
            headers.append('Potential Energy (kJ/mole)')
        if self._kineticEnergy:
            headers.append('Kinetic Energy (kJ/mole)')
        if self._totalEnergy:
            headers.append('Total Energy (kJ/mole)')
        if self._volume:
            headers.append('Box Volume (nm^3)')
        if self._elapsedTime:
            headers.append('Elapsed Time (s)')
        if self._speed:
            headers.append('Speed (ns/day)')
        return headers

    def __del__(self):
        if self._openedFile:
            self._out.close()


    