using Suppressor
using ChebyshevDFT
      
function run_go(Z)


    TO = zeros(0,11);

    if Z <= 2
        core = 0
        name = " "
        inds = 1
        indp = 1
        indd = -1
        attype = :s
    elseif Z <= 4
        core = 2
        name = "He"
        inds = 2
        indp = 1
        indd = -1
        attype = :s
    elseif Z <= 10
        core = 2
        name = "He"
        inds = 2
        indp = 1
        indd = -1
        attype = :p
    elseif Z <= 12
        core = 10
        name = "Ne"
        inds = 3
        indp = 2
        indd = -1
        attype = :s
    elseif Z <= 18
        core = 10
        name = "Ne"
        inds = 3
        indp = 2
        indd = -1
        attype = :p
    elseif Z <= 20
        core = 18
        name = "Ar"
        inds = 4
        indp = 3
        indd = 1
        attype = :s
    elseif Z <= 30
        core = 18
        name = "Ar"
        inds = 4
        indp = 3
        indd = 1
        attype = :d
    elseif Z <= 36
        core = 30
        name = "Zn"
        inds = 4
        indp = 3
        indd = 1
        attype = :p
    elseif Z <= 38
        core = 36
        name = "Kr"
        inds = 5
        indp = 4
        indd = 2
        attype = :s
    elseif Z <= 48
        core = 36
        name = "Kr"
        inds = 5
        indp = 4
        indd = 2
        attype = :d
    elseif Z <= 54
        core = 48
        name = "Cd"
        inds = 5
        indp = 4
        indd = 2
        attype = :p
    elseif Z <= 56
        core = 54
        name = "Kr"
        inds = 6
        indp = 5
        indd = 3
        attype = :s
    elseif Z <= 80
        core = 54
        name = "Xe"
        inds = 6
        indp = 5
        indd = 3
        attype = :d
    elseif Z <= 86
        core = 80
        name = "Hg"
        inds = 6
        indp = 5
        indd = 3
        attype = :p
    end

    println([core, name,inds, indp,indd,attype])


    
    if attype == :p

        other = (Z - core - 3)/2/2
        
        rho_start = missing


        
        for m = 0.5:.5/4:1.0;
            t = """
            $name
            1 0 0 $m $(1-m)  
            1 1 1 $other $other
            1 1 0 $other $other
            1 1 -1 $other $other
                    """; 
            println("t")
            println(t)
            energyN, converged, vals_rS, vectsN, rho_LMN, rall_rsN, aqqqN, rhor2N, vlda_LMN, drho_LMN,D1N,vaN  =  ChebyshevDFT.SCF.DFT_spin_l_grid_LM(fill_str = t, Z=Z, N=80, Rmax=35, niters=80, exc = [:gga_x_pbe_sol, :gga_c_pbe_sol], rho_init = rho_start);
            
            rho_start = rho_LMN[:,1,1,1] + rho_LMN[:,2,1,1]
            
            mag = 2*(m-0.5)

            if converged 
                if indd > 0
                    TO = [TO; mag vals_rS[inds,1,1,1] vals_rS[inds,2,1,1] vals_rS[indp,1,2,1] vals_rS[indp,2,2,1] vals_rS[indp,1,1,2] vals_rS[indp,2,1,2] vals_rS[indd,1,3,1] vals_rS[indd,2,3,1] vals_rS[indd,1,1,4] vals_rS[indd,2,1,4]]
                else
                    TO = [TO; mag vals_rS[inds,1,1,1] vals_rS[inds,2,1,1] vals_rS[indp,1,2,1] vals_rS[indp,2,2,1] vals_rS[indp,1,1,2] vals_rS[indp,2,1,2] 0.0 0.0 0.0 0.0]
                end
            end
        end

        Wss =  TO[:,1] \ (TO[:,2] - TO[:,3])  
        Wsp  = TO[:,1] \ (TO[:,4] - TO[:,5])  
        Wsd =  TO[:,1] \(TO[:,8] - TO[:,9]) 
        Wpp =  0.0
        Wpd =  0.0
        WppD = 0.0
        
        Xss =  TO[:,1] \(TO[:,2] + TO[:,3] .- TO[1,2]*2)  
        Xsp =  TO[:,1] \(TO[:,4] + TO[:,5] .- TO[1,4]*2)  
        Xsd =  TO[:,1] \  (TO[:,8] + TO[:,9] .- TO[1,8]*2)  
        Xpp =  0.0
        XppD =  0.0
        Xpd = 0.0
        
        Wdd = 0.0
        WddD = 0.0
        
        Xdd = 0.0
        XddD = 0.0
        
        
#        Wsp =  TO[:,1] \ (TO[:,2] - TO[:,3])  
#        WppD = TO[:,1] \ (TO[:,4] - TO[:,5])  
#        Wpp =  TO[:,1] \(TO[:,6] - TO[:,7])  
#        Wpd =  TO[:,1] \(TO[:,8] - TO[:,9]) 
        
#        Xsp =  TO[:,1] \(TO[:,2] + TO[:,3] .- TO[1,2]*2)  
#        XppD = TO[:,1] \(TO[:,4] + TO[:,5] .- TO[1,4]*2)  
#        Xpp =  TO[:,1] \(TO[:,6] + TO[:,7] .- TO[1,6]*2)  
#        Xpd =  TO[:,1] \  (TO[:,8] + TO[:,9] .- TO[1,8]*2)  

#        Wss = 0.0
#        Xss = 0.0
        
#        Wdd = 0.0
#        WddD = 0.0
        
#        Xdd = 0.0
#        XddD = 0.0

#        Wsd = 0.0
#        Xsd = 0.0
        
    elseif attype == :s



        println("attype s")
        
        rho_start = missing
        
        for m = 0.5:.5/4:1.0;
            if Z == 1.0 || Z == 2.0
                t = """
                    1 0 0 $m $(1-m)
                    1 1 0 0.0 0.0
                    1 1 -1 0.0 0.0
                    1 1 1 0.0 0.0
                    """;
            elseif Z < 13
                t = """
                    $name
                    1 0 0 $m $(1-m)
                    1 1 0 0.0 0.0
                    1 1 -1 0.0 0.0
                    1 1 1 0.0 0.0
                    """;
            else
                t = """
                    $name
                    1 0 0 $m $(1-m)
                    1 1 0 0.0 0.0
                    1 1 -1 0.0 0.0
                    1 1 1 0.0 0.0
                1 2 2 0.0 0.0
                1 2 1 0.0 0.0 
                1 2 0  0.0 0.0
                1 2 -1  0.0 0.0
                1 2 -2  0.0 0.0
                    """;
            end                
            energyN, converged, vals_rS, vectsN, rho_LMN, rall_rsN, aqqqN, rhor2N, vlda_LMN, drho_LMN,D1N,vaN  =  ChebyshevDFT.SCF.DFT_spin_l_grid_LM(fill_str = t, Z=Z, N=100, Rmax=35, niters=80, exc = [:gga_x_pbe_sol, :gga_c_pbe_sol], rho_init = rho_start);
            
            rho_start = rho_LMN[:,1,1,1] + rho_LMN[:,2,1,1]
            
            mag = 2*(m-0.5)

            if converged 
                if indd > 0
                    TO = [TO; mag vals_rS[inds,1,1,1] vals_rS[inds,2,1,1] vals_rS[indp,1,2,1] vals_rS[indp,2,2,1] vals_rS[indp,1,1,2] vals_rS[indp,2,1,2] vals_rS[indd,1,3,1] vals_rS[indd,2,3,1] vals_rS[indd,1,1,4] vals_rS[indd,2,1,4]]
                else
                    TO = [TO; mag vals_rS[inds,1,1,1] vals_rS[inds,2,1,1] vals_rS[indp,1,2,1] vals_rS[indp,2,2,1] vals_rS[indp,1,1,2] vals_rS[indp,2,1,2] 0.0 0.0 0.0 0.0]
                end
            end


        end
        

        Wss =  TO[:,1] \ (TO[:,2] - TO[:,3])  
        Wsp  = TO[:,1] \ (TO[:,4] - TO[:,5])  
        Wsd =  TO[:,1] \(TO[:,8] - TO[:,9]) 
        Wpp =  0.0
        Wpd =  0.0
        WppD = 0.0
        
        Xss =  TO[:,1] \(TO[:,2] + TO[:,3] .- TO[1,2]*2)  
        Xsp =  TO[:,1] \(TO[:,4] + TO[:,5] .- TO[1,4]*2)  
        Xsd =  TO[:,1] \  (TO[:,8] + TO[:,9] .- TO[1,8]*2)  
        Xpp =  0.0
        XppD =  0.0
        Xpd = 0.0
        
        Wdd = 0.0
        WddD = 0.0
        
        Xdd = 0.0
        XddD = 0.0


    elseif attype == :d

        other = (Z - core - 1 - 14 - 0.05)/2/5 

        if other >= 1.0
            other = 0.95
        end
        if other < 0
            other = 0.0
        end

        
        rho_start = missing
        energyN, converged, vals_rS, vectsN, rho_LMN, rall_rsN, aqqqN, rhor2N, vlda_LMN, drho_LMN,D1N,vaN  =  ChebyshevDFT.SCF.DFT_spin_l_grid_LM(Z=Z, N=100, Rmax=35, niters=90, exc = [:gga_x_pbe_sol, :gga_c_pbe_sol], mix=0.5);
        rho_start = rho_LMN[:,1,1,1]
        
        for m = 0.5:.5/4:1.0;
            flush(stdout)
            if Z <= 54
                t = """
                $name
                1 0 0 $m $(1-m)
                1 2 2 $other $other
                1 2 1 $other $other
                1 2 0 $other $other
                1 2 -1 $other $other
                1 2 -2 $other $other
                """;
            else
                t = """
                $name
                1 3 3 1.0 1.0
                1 3 2 1.0 1.0
                1 3 1 1.0 1.0
                1 3 0 1.0 1.0
                1 3 -1 1.0 1.0
                1 3 -2 1.0 1.0
                1 3 -3 1.0 1.0
                1 0 0 $m $(1-m)
                1 2 2 $other $other
                1 2 1 $other $other
                1 2 0 $other $other
                1 2 -1 $other $other
                1 2 -2 $other $other
                """;
            end                
            
            println("t")
            println(t)
            energyN, converged, vals_rS, vectsN, rho_LMN, rall_rsN, aqqqN, rhor2N, vlda_LMN, drho_LMN,D1N,vaN  =  ChebyshevDFT.SCF.DFT_spin_l_grid_LM(fill_str = t, Z=Z, N=100, Rmax=35, niters=100, exc = [:gga_x_pbe_sol, :gga_c_pbe_sol], rho_init = rho_start, mix = 0.4);
            
            rho_start = rho_LMN[:,1,1,1] + rho_LMN[:,2,1,1]
            
            mag = 2*(m-0.5)

            if converged 
                if indd > 0
                    TO = [TO; mag vals_rS[inds,1,1,1] vals_rS[inds,2,1,1] vals_rS[indp,1,2,1] vals_rS[indp,2,2,1] vals_rS[indp,1,1,2] vals_rS[indp,2,1,2] vals_rS[indd,1,3,1] vals_rS[indd,2,3,1] vals_rS[indd,1,1,4] vals_rS[indd,2,1,4]]
                else
                    TO = [TO; mag vals_rS[inds,1,1,1] vals_rS[inds,2,1,1] vals_rS[indp,1,2,1] vals_rS[indp,2,2,1] vals_rS[indp,1,1,2] vals_rS[indp,2,1,2] 0.0 0.0 0.0 0.0]
                end
            end
        end
        

        Wss =  TO[:,1] \ (TO[:,2] - TO[:,3])  
        Wsp  = TO[:,1] \ (TO[:,4] - TO[:,5])  
        Wsd =  TO[:,1] \(TO[:,8] - TO[:,9]) 
        Wpp =  0.0
        Wpd =  0.0
        WppD = 0.0
        
        Xss =  TO[:,1] \(TO[:,2] + TO[:,3] .- TO[1,2]*2)  
        Xsp =  TO[:,1] \(TO[:,4] + TO[:,5] .- TO[1,4]*2)  
        Xsd =  TO[:,1] \  (TO[:,8] + TO[:,9] .- TO[1,8]*2)  
        Xpp =  0.0
        XppD =  0.0
        Xpd = 0.0
        
        Wdd = 0.0
        WddD = 0.0
        
        Xdd = 0.0
        XddD = 0.0

        
    end
    
    println("FITOUT $Z $Wss $Wsp $Wsd $Wpp $Wpd $Wdd $WppD $WddD     $Xss $Xsp $Xsd $Xpp $Xpd $Xdd $XppD $XddD     ")

    return TO
    
end


#for Z = cat(28:30, 40:48, 72:80, dims=1)
for Z = 72:80
    run_go(Z)
end
