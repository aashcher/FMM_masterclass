
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                         %% DEMO SIMULATIONS %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % photonic crystal slab:
phc_slab.period = 1.2;
phc_slab.thickness = 0.24;
phc_slab.period_fill_factors = [0.45, 0.55];
phc_slab.period_epsilons = [2.25, 1.0];

  % diffraction problem:
angle_inc = 30;
wavelength = 0.6328;
polarization = 's';
epsilon_substrate = 2.25;
epsilon_superstrate = 1;

number_harmonics = 15;
ic = ceil(number_harmonics/2); % zero harmonic index

kx0 = sin(pi*angle_inc/180); % incident plane wave wavevector in-plane projection normalized by the vacuum wavenumber
kG = wavelength / phc_slab.period; % reciprocal lattice vector normalized by the vacuum wavenumber

  % fill the structure difining the grating diffraction problem:
grating_diffraction_problem.number_harmonics = number_harmonics; % number of Fourier harmonics to consider in calculations
grating_diffraction_problem.ic = ic; % zero harmonic index
grating_diffraction_problem.kh = 2 * pi * phc_slab.thickness / wavelength; % normalized grating slab thickness
grating_diffraction_problem.kx0 = kx0;
grating_diffraction_problem.kG = kG;
grating_diffraction_problem.kx = get_kx(number_harmonics, kx0, kG); % pre-calculate reciprocal lattice vectors
grating_diffraction_problem.polarization = polarization;
grating_diffraction_problem.epsilon_substrate = epsilon_substrate;
grating_diffraction_problem.epsilon_superstrate = epsilon_superstrate;

  % calculate Fourier amplitudes of the permittivity and inverse permittivity periodic functions:
F = get_phc_epsilon_Fourier(grating_diffraction_problem, phc_slab);
  % calculate grating S-matrix and eigen states of a corresponding photonic crystal:
[S, beta, M, EV, HV] = get_smatrix_phcslab(grating_diffraction_problem, F);
  % fill an amplitude vector of an incident plane wave:
Vin = plane_wave(grating_diffraction_problem, 'from above');
  % calculate an amplitude vector of for the diffracted field:
Vout = smatrix_diffract(S, Vin);
  % calculate a vector of diffraction efficiencies:
Veff = get_diffraction_efficiencies(Vin, Vout, grating_diffraction_problem);

fprintf('diffraction efficiensies:\n')
for order = -2:2
	fprintf('order %d: t = %f,   r = %f\n', order, real(Veff(dindex(order, grating_diffraction_problem),1)), ...
																									real(Veff(dindex(order, grating_diffraction_problem),2)));	
end

  % calculate and plot a local field in the grating region:
nx = 100;
nz = 100;
  % calculate the field inside the grating:
[MFx, MFy, MFz] = get_local_field(grating_diffraction_problem, phc_slab, ...
									Vin, beta, M, EV, HV, F, nx, nz);
  % calculate the field below the grating:
V = zeros(number_harmonics, 2);
V(:,2) = Vout(:,1);
[MFsubx, MFsuby, MFsubz] = get_layer_field(V, grating_diffraction_problem, grating_diffraction_problem.kh, epsilon_substrate, nx, nz);
  % calculate the field above the grating:
V(:,1) = Vout(:,2);
V(:,2) = Vin(:,2) * exp(-1i*sqrt(epsilon_superstrate - kx0^2)*phc_slab.thickness*2.0*pi/wavelength);
[MFsupx, MFsupy, MFsupz] = get_layer_field(V, grating_diffraction_problem, grating_diffraction_problem.kh, epsilon_superstrate, nx, nz);
% concantenate the fields:
MFx = [MFsubx, MFx, MFsupx];
MFy = [MFsuby, MFy, MFsupy];
MFz = [MFsubz, MFz, MFsupz];

plot_x = (phc_slab.period / nx)*((1:nx)-1);
plot_z = (phc_slab.thickness / nz)*((1:3*nz)-0.5-nz);
[X, Z] = meshgrid(plot_x, plot_z);
surf(X, Z, abs(MFy)', 'EdgeColor', 'none');
view(2);
colorbar;

return;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                            %% FUNCTIONS %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% common functions: %%
%{
	get_kx: calculate reciprocal lattice vectors
	input:
		n: number of reciprocal lattice vectors to calculate
		kx0: Bloch wavenumber, normalized by the vaccum wavenumber
		kG: reciprocal lattice basis vector, normalized by the vaccum wavenumber
	output:
		kx: a column of reciprocal lattice vectors, normalized by the vacuum 
				wavenumber
%}
function kx = get_kx(n, kx0, kG)
	ind = linspace(1,n,n);
	kx = kx0 + kG*(ind - ceil(n/2));
end

%{
	get_kz: calculate vertical wavevector projections in a homogeneous medium 
					for a given reciprocal lattice
	input:
		kx: column of reciprocal lattice vectors, normalized by the vacuum 
				wavenumber
		eps: permittivity of the medium
	output:
		kz: column of vertical, normalized by the vacuum wavenumber
%}
function kz = get_kz(kx, eps)
		% plane wave dispersion equation
	kz = sqrt(eps - kx.^2);
		% check for the half plane
	ind = angle(kz) < -1e-12;
	kz(ind) = -kz(ind);
end

%{
	dindex: map diffraction order number to amlitude vector index
	inout:
		order: diffraction order number
		problem: structure defining the diffraction problem
	output:
		ind: index in the amplitude vector
%}
function ind = dindex(order, problem)
	ind = problem.ic + order;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% S-matrix functions: %%
%{
	get_smatrix_layer: calculate S-matrix of a homogeneous layer
	input parameters:
		n: number of reciprocal lattice vectors
		kx0: Bloch wavenumber, normalized by the vaccum wavenumber
		kG: reciprocal lattice basis vector, normalized by the vaccum wavenumber
		kh: layer thikness multiplied by the vaccum wavenumber
		eps: permittivity of the layer
	output parameters:
		S: S-matrix
%}
function S = get_smatrix_layer(n, kx0, kG, kh, eps)
	kz = get_kxz(get_kx(n, kx0, kG), eps);
	S = zeros(n,n,2,2);
	S(:,:,1,2) = diag( exp((1i*kh)*kz) );
	S(:,:,2,1) = S(:,:,1,2);
end

%{
	get_smatrix_interface: calculate S-matrix of an interface z=0 between two
													homogeneous layers
	input parameters:
		n: number of reciprocal lattice vectors
		kx0: Bloch wavenumber, normalized by the vaccum wavenumber
		kG: reciprocal lattice basis vector, normalized by the vaccum wavenumber
		eps1: permittivity of the medium below the interface
		eps2: permittivity of the medium above the interface
		pol: polarization, either 's' or 'p'
	output parameters:
		S: S-matrix
%}
function S = get_smatrix_interface(n, kx0, kG, eps1, eps2, pol)
	kx = get_kx(n, kx0, kG);
	kz1 = get_kxz(kx, eps1);
	kz2 = get_kxz(kx, eps2);
	if strcmp(pol,'p')
		kz1 = (1/eps1) * kz1;
		kz2 = (1/eps2) * kz2;
	end

	S = zeros(n,n,2,2);
	S(:,:,1,1) = diag( (kz1-kz2)./(kz1+kz2) );
	S(:,:,2,1) = 1 + S(:,:,1,1);
	S(:,:,2,2) = -S(:,:,1,1);
	S(:,:,1,2) = 1 + S(:,:,2,2);
end

%{
	smatrix_diffract: calculate amplitude vector for outgouing waves for a
										given S-matrix and a vector of incoming amplitudes
	input:
		S: S-matrix of size (n,n,2,2)
		Vin: input amplitude vector of size (n,2)
	output:
		Vout: output amplitude vector of size (n,2)
%}
function Vout = smatrix_diffract(S, Vin)
	Vout = Vin;
	Vout(:,1) = S(:,:,1,1)*Vin(:,1) + S(:,:,1,2)*Vin(:,2);
	Vout(:,2) = S(:,:,2,1)*Vin(:,1) + S(:,:,2,2)*Vin(:,2);
end

%{
	smatrix_emit: calculate amplitude vector for emitted waves for a
								source placed between layers with given S-matrices
	input:
		S1: S-matrix of a layer below the source, S-matrix size is (n,n,2,2)
		S2: S-matrix of a layer above the source, S-matrix size is (n,n,2,2)
		Vsrc: vector of source wave amplitudes of size (n,2)
	output:
		Vout: output amplitude vector of size (n,2)
%}
function Vemi = smatrix_emit(S1, S2, Vsrc)
	n = size(Vsrc,1);
	Vemi = Vsrc;
	Vemi(:,1) = ( S1(:,:,1,2) / (eye(n) - S2(:,:,1,1)*S1(:,:,2,2)) ) * ...
							( S2(:,:,1,1) * Vsrc(:,1) + Vsrc(:,2) );
	Vemi(:,2) = ( S2(:,:,2,1) / (eye(n) - S1(:,:,2,2)*S2(:,:,1,1)) ) * ...
							( Vsrc(:,1) + S1(:,:,2,2) * Vsrc(:,2) );
end

%{
	smatrix_multiply: calculate S-matrix of a composite structure consisting
										of two sub-structures with given S-matrices
	input:
		S1: S-matrix of a lower structure
		S2: S-matrix of an upper structure
	output:
		S: S-matrix of the composition of the two structures
%}
function S = smatrix_multiply(S1, S2)
	n = size(S1,1);
	S = zeros(n,n,2,2);

	tmp = -S2(:,:,1,1) * S1(:,:,2,2);
	tmp(1:n+1:end) = tmp(1:n+1:end) + 1;
	tmp = S1(:,:,1,2) / tmp;
	S(:,:,1,2) = tmp * S2(:,:,1,2);
	S(:,:,1,1) = S1(:,:,1,1) + tmp * S2(:,:,1,1) * S1(:,:,2,1);

	tmp = -S1(:,:,2,2) * S2(:,:,1,1);
	tmp(1:n+1:end) = tmp(1:n+1:end) + 1;
	tmp = S2(:,:,2,1) / tmp;
	S(:,:,2,1) = tmp * S1(:,:,2,1);
	S(:,:,2,2) = S2(:,:,2,2) + tmp * S1(:,:,2,2) * S2(:,:,1,2);
end

%{
	plane_wave: get amplitude vector for a plane wave
	input:
		problem: structure which defines the grating diffraction problem
		direction: propagation direction, either 'from above' or 'from below'
	output:
		S: S-matrix of the composition of the two structures
%}
function V = plane_wave(problem, direction)
	n = problem.number_harmonics;
	ic = problem.ic;
	V = zeros(n,2);
	if strcmp(direction, 'from above')
		V(ic,2) = 1;
	elseif strcmp(direction, 'from below')
		V(ic,1) = 1;
	else
		error('plane_wave: unknown direction')
	end
end

%% Fourier Modal Method functions: %%
%{
	get_tmatrix_phc_interface: calculate T-matrix of an interface between a 
			semi-infinite 1D photonic crystal and a homogeneous medium
	input parameters:
		n: number of Fourier harmonics
		EV: Fourier amplitude matrix of the electric field eigenvectors of the photonic crystal
		HV: Fourier amplitude matrix of the magnetic field eigenvectors of the photonic crystal
		kz: vector of plane wave z-projections
		eps: permittivity of the medium above the photonic crystal
		pol: polarization, either 's' or 'p'
	output parameters:
		T: interface T-matrix
%}
function T = get_tmatrix_phc_interface(n, EV, HV, kz, eps, pol)
	T = zeros(2*n,2*n);
	ib1 = 1:n;
	ib2 = n+1:2*n;
	if strcmp(pol, 's')
		ikz = transpose(0.5./kz);
		T(ib1,ib1) = -HV.*ikz;
		T(ib2,ib1) = 0.5*EV - T(ib1,ib1);
		T(ib1,ib1) = T(ib1,ib1) + 0.5*EV;
		T(ib1,ib2) = T(ib2,ib1);
		T(ib2,ib2) = T(ib1,ib1);
	elseif strcmp(pol, 'p')
		eikz = transpose((0.5*eps)./kz);
		T(ib1,ib1) = EV.*eikz;
		T(ib2,ib1) = 0.5*HV - T(ib1,ib1);
		T(ib1,ib1) = T(ib1,ib1) + 0.5*HV;
		T(ib1,ib2) = -T(ib2,ib1);
		T(ib2,ib2) = -T(ib1,ib1);
	end
end

%{
	get_smatrix_phcslab: calculate S-matrix of a 1D photonic crystal slab
	input parameters:
		problem: structure, which defines the grating diffraction problem
		FM: matrix containing Fourier amplitude vectors of the permittivity and
				inverse permittivity functions of a photonic crystal
		eps1, eps2: permittivities of substrate and superstrate media
	output parameters:
		S: S-matrix
%}
function [S, beta, M2, EV, HV] = get_smatrix_phcslab(problem, FM)
	n = problem.number_harmonics;
		% reciprocal lattice and vertical wavevector projections
	kx = problem.kx; %kx = get_kx(n, kx0, kG);
	kz1 = get_kz(kx, problem.epsilon_substrate);
	kz2 = get_kz(kx, problem.epsilon_superstrate);
	  % solve the eigenvalue problem for an infinite photonic crystal:
	ME = toeplitz(FM(n:2*n-1,1),FM(n:-1:1,1)); % Toeplitz matrix of the permittivity function Fourier amplitudes
	if (strcmp(problem.polarization, 's')) % s polarization
		ME(1:n+1:end) = ME(1:n+1:end) - (kx.^2);
		[EV, beta] = eig(ME, 'vector');
		beta = transpose(beta);
			% sort eigenvalues and eigenvectors:
		[beta, bind] = sort(beta, 'descend', 'ComparisonMethod', 'real');
		EV(:, 1:n) = EV(:, bind);
			% calculate mode propagation constants and check the branch of the
			% square root:
		beta = sqrt(beta);
		ind = angle(beta) < -1e-12;
		beta(ind) = -beta(ind);
		disp(beta)
			% magnetic field for the calculated Bloch modes (Hx):
		HV = -EV.*beta;
	else % p polarization
		MU = eye(n) / toeplitz(FM(n:2*n-1,2),FM(n:-1:1,2)); % Toeplitz matrix of the inverse permittivity function Fourier amplitudes
		ME = -(diag(kx) / ME).*kx;
		ME(1:n+1:end) = ME(1:n+1:end) + 1;
		[EV,beta] = eig(ME*MU, 'vector');
		beta = transpose(beta);
			% sort eigenvalues and eigenvectors:
		[beta,bind] = sort(beta,'descend','ComparisonMethod','real');
		EV(:,1:n) = EV(:,bind);
			% calculate mode propagation constants and check the branch of the
			% square root:
		beta = sqrt(beta);
		ind = angle(beta) < -1e-7;
		beta(ind) = -beta(ind);
			% magnetic field for the calculated Bloch modes (Hx):
		HV = (MU*EV)./beta;
	end
	  % calculate T-matrices of the interfaces between the photonic crystal 
		% and surrounding media:
	TS = get_tmatrix_phc_interface(n, EV, HV, kz1, problem.epsilon_substrate, problem.polarization); % susbtrate-grating T-matrix
	TC = get_tmatrix_phc_interface(n, EV, HV, kz2, problem.epsilon_superstrate, problem.polarization); % grating-superstrate T-matrix
	%disp(TS(1:n,1:n))
		% combine blocks of the T-matrices
		% auxiliary block indices:
	ib1 = 1:n;
	ib2 = n+1:(2*n);
	bexp = exp((1i*problem.kh)*beta);
	M1 = zeros(2*n,2*n);
	M2 = zeros(2*n,2*n);
	M1(ib1,ib1) = TS(ib2,ib1);
	M1(ib1,ib2) = TS(ib2,ib2).*bexp;
	M1(ib2,ib1) = TC(ib1,ib1).*bexp;
	M1(ib2,ib2) = TC(ib1,ib2);
	M2(ib1,ib1) = TS(ib1,ib1);
	M2(ib1,ib2) = TS(ib1,ib2).*bexp;
	M2(ib2,ib1) = TC(ib2,ib1).*bexp;
	M2(ib2,ib2) = TC(ib2,ib2);
		% final S-matrix
	M1 = M1/M2;
	S = zeros(n,n,2,2);
	S(:,:,1,1) = M1(ib1,ib1);
	S(:,:,1,2) = M1(ib1,ib2);
	S(:,:,2,1) = M1(ib2,ib1);
	S(:,:,2,2) = M1(ib2,ib2);
end

%{
	get_phc_epsilon_Fourier: calculate Fourier amplitude vectors of 1D 
					photonic crystal permittivity and inverse permittivity functions
	input parameters:
		problem: structure, which defines a diffraction problem formulation
						 for the Fourier Modal Method
		phc_slab: structure, which defines parameters of 1D photonic crystal
							slab
	output parameters:
		F: two-column matrix containing Fourier amplitude vectors of the
			 permittivity and inverse permittivity functions
%}
function F = get_phc_epsilon_Fourier(problem, phc_slab)
	n = problem.number_harmonics;
	ff = phc_slab.period_fill_factors;
	eps = phc_slab.period_epsilons;

		% check for consistency of the parameters:
	if (length(ff) ~= length(eps)) || (abs(sum(ff)-1) > 1e-12)
		error("get_phc_epsilon_Fourier: input error");
	end

	pos = -0.5 + 0.5*ff;
	pos(2:end) = pos(2:end) + cumsum(ff(1:end-1));

	F = zeros(2*n-1, 2);
	ind = transpose(linspace(1, n-1, n-1));

	for k = 1:length(ff)
		ifun = sin(ind*pi*ff(k))./(ind*pi);
		te = exp(-(2*pi*1i*pos(k))*ind);
			% zero harmonics:
		F(n,1) = F(n,1) + eps(k)*ff(k);
		F(n,2) = F(n,2) + ff(k)/eps(k);
			% non-zero harmonics:
		tmp = eps(k)*ifun;
		F(n+1:2*n-1,1) = F(n+1:2*n-1,1) + tmp.*te;
		F(n-1:-1:1,1) = F(n-1:-1:1,1) + tmp.*conj(te);
		tmp = (1/eps(k))*ifun;
		F(n+1:2*n-1,2) = F(n+1:2*n-1,2) + tmp.*te;
		F(n-1:-1:1,2) = F(n-1:-1:1,2) + tmp.*conj(te);
	end
end

%{
	get_phc_iepsilon: calculate inverse permittivity values in a period of
										a given photonic crystal
	input:
		x_points: array of points on a period, should be in the range [0.5,0.5]
		phc_slab: structure, which defines parameters of 1D photonic crystal
							slab
	output:
		x_eps: values of inverse permittivyt in x_points
%}
function x_eps = get_phc_iepsilon(nx, phc_slab)
	x_points = linspace(0,nx-1,nx) / nx
	x_points(x_points > 0.5) = x_points(x_points > 0.5) - 1.0

	ff = phc_slab.period_fill_factors;
	epss = phc_slab.period_epsilons;

	bounds = [cumsum(ff)-ff-0.5, 0.5]; % bound positions between layers

	x_eps = x_points;
	for i = 1:numel(ff)
		ind = (x_points >= bounds(i)) & (x_points <= bounds(i+1));
		x_eps(ind) = 1/epss(i);
	end
end

%{
	get_diffraction_efficiencies: calculate T-matrix of an interface between a 
			semi-infinite 1D photonic crystal and a homogeneous medium
	input parameters:
		n: number of Fourier harmonics
		EV: Fourier amplitude matrix of the electric field eigenvectors of the photonic crystal
		HV: Fourier amplitude matrix of the magnetic field eigenvectors of the photonic crystal
		kz: vector of plane wave z-projections
		eps: permittivity of the medium above the photonic crystal
		pol: polarization, either 's' or 'p'
	output parameters:
		T: interface T-matrix
%}
function Veff = get_diffraction_efficiencies(Vin, Vout, problem)
		% reciprocal lattice and vertical wavevector projections
	kx = problem.kx;
	kz1 = (get_kz(kx, problem.epsilon_substrate)).';
	kz2 = (get_kz(kx, problem.epsilon_superstrate)).';
	if (strcmp(problem.polarization,'p'))
		kz1 = kz1 / problem.epsilon_substrate;
		kz2 = kz2 / problem.epsilon_superstrate;
	end

	Veff = zeros(size(Vin));
	Pin = sum( abs(Vin(:,1).^2).*real(kz1) + abs(Vin(:,2).^2).*real(kz2) );
	Veff(:,1) = abs(Vout(:,1).^2).*real(kz1);
	Veff(:,2) = abs(Vout(:,2).^2).*real(kz2);

	if (abs(Pin) > 1e-15)
		Veff = Veff/Pin;
	end
end

%{
	get_local_field: calculate electromagnetic fields in the coordinate space
									inside a grating slab
	input parameters:
		problem: structure defining a grating diffraction problem
		phc_slab: structure defining a photonic crystal slab
		Vin: Fourier amplitude vector of an incident field
		beta: vector of propagation constants for the slab calculated by the function get_smatrix_phcslab
		Ms: matrix which couples the incident amplitudes below the slab with mode amplitudes inside the slab, 
				calculated by the function get_smatrix_phcslab
		EV, HV: Fourier amplitude metrices of the electric and magnetic fields 
						of the slab modes
		FE: matrix containig the Fourier amplitude vectors of the slab permittivity 
				and inverse permittivity functions
		nx: number of points along a single period where to evaluate the fields
		nz: number of points along the slab depth where to evaluate the fields
	output parameters:
		MFx, MFy, MFz: matrices of size (nx, nz) containing the field values at
  gigen coordinate space points
									H_x, E_y, H_z in case of s polarization
									E_x, H_y, E_z in case of p polarization
%}
function [MFx, MFy, MFz] = get_local_field(problem, phc_slab, Vin, beta, Ms, EV, HV, FE, nx, nz)
	n = numel(beta);
		% mode amplitudes
	Vmod = reshape(Ms\reshape(Vin,[],1), [n,2]);

	MFx = zeros(nx, nz);
	MFy = zeros(nx, nz);
	MFz = zeros(nx, nz);

	texp = (exp(2.0*1i*(problem.kx0/problem.kG/nx - ceil(nx/2)/nx) * linspace(0,nx-1,nx))).';

	if strcmp(problem.polarization, 's')
		for i = 1:nz
			bexp1 = exp((1i*((i-0.5)/nz) * problem.kh) * (beta.'));
			bexp2 = exp((1i*((nz-i+0.5)/nz) * problem.kh) * (beta.'));
			Vp = Vmod(:,1) .* bexp1;
			Vm = Vmod(:,2) .* bexp2;
			MFy(:,i) = nx * ifft(EV*(Vp + Vm), nx); % E_y
			MFx(:,i) = nx * ifft(HV*(Vp - Vm), nx); % H_x
			MFz(:,i) = nx * ifft(diag(problem.kx)*EV*(Vp + Vm), nx); % H_z

			MFy(:,i) = MFy(:,i) .* texp;
			MFx(:,i) = MFx(:,i) .* texp;
			MFz(:,i) = MFz(:,i) .* texp;
		end
	elseif strcmp(problem.polarization, 'p')
		ME = eye(n) / toeplitz(FE(n:2*n-1,1), FE(n:-1:1,1));
		MU = eye(n) / toeplitz(FE(n:2*n-1,2), FE(n:-1:1,2));
		x_points = linspace(-0.5, 0.5-1.0/nx, nx);
		x_ieps = get_phc_iepsilon(x_points, phc_slab);
		for i = 1:nz
			bexp1 = exp((1i*((i-0.5)/nz) * problem.kh) * (beta.'));
			bexp2 = exp((1i*((nz-i+0.5)/nz) * problem.kh) * (beta.'));
			Vp = Vmod(:,1) .* bexp1;
			Vm = Vmod(:,2) .* bexp2;
			MFy(:,i) = nx * ifft(HV*(Vp - Vm), nx); % H_y
			MFx(:,i) = nx * (x_ieps).' .* ifft(MU*EV*(Vp + Vm), nx); % E_x
			MFz(:,i) = -nx * ifft(ME*diag(problem.kx)*HV*(Vp - Vm), nx); % E_z

			MFy(:,i) = MFy(:,i) .* texp;
			MFx(:,i) = MFx(:,i) .* texp;
			MFz(:,i) = MFz(:,i) .* texp;
		end
	end
end

%{
	get_near_field: calculate electromagnetic fields in the coordinate space
				 in a homogeneous layer
	input parameters:
		V: Fourier amplitude vector of the incident field: V(:,1) are
  amplitudes of up-propagating plane waves at the bottom side of the layer,
  and V(:,2) are amplitudes of down-propagating plane waves at the to side
  of the layer
		problem: structure defining a grating diffraction problem
		kh: layer thickness normalized by the wavelength
		epsilon: layer dielectric permittivity
		nx: number of points along a single period where to evaluate the fields
		nz: number of points along the slab depth where to evaluate the fields
	output parameters:
		MFx, MFy, MFz: matrices of size (nx, nz) containing the field values
									H_x, E_y, H_z in case of s polarization
									E_x, H_y, E_z in case of p polarization
%}
function [MFx, MFy, MFz] = get_layer_field(V, problem, kh, epsilon, nx, nz)
	kx = problem.kx;
	kz = (get_kz(kx, epsilon)).';
	if strcmp(problem.polarization, 's')
		cz = 1;
	elseif strcmp(problem.polarization, 'p')
		cz = -1/epsilon;
	end
	kzp = abs(cz) * kz;

	MFx = zeros(nx, nz);
	MFy = zeros(nx, nz);
	MFz = zeros(nx, nz);

	kdh = kh/nz;

	for k = 1:nz
		z = (k-0.5)*kdh;
		Vm = V(:,2).*exp(1i*kz*(kh-z));
		Vp = V(:,1).*exp(1i*kz*z);
		MFy(:,k) = nx * ifft((Vp + Vm), nx);
		MFx(:,k) = nx * ifft(kzp.*(Vm - Vp), nx);
		MFz(:,k) = cz * nx * ifft(diag(kx)*(Vp + Vm), nx);
	end
end

%% end of file %%


