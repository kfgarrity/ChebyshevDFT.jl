using Optim

function asdf()

    T = 2.0 * I(10) - diagm(1 => [1,1,1,1,1,1,1,1,1])  -  diagm(-1 => [1,1,1,1,1,1,1,1,1]) 
    V = diagm([4,3,2,1,0,0,1,2,3,4.0]*1.0)
    H = T + V

    vals, vects = eigen(H)
    n = deepcopy(vects[:,1].^2 + vects[:,2].^2)

    constval = 1000.0
    function f(v)
        H = T + diagm(v)
        vals, vects = eigen(H)
        n_temp = vects[:,1].^2 + vects[:,2].^2
        ret = sum((n - n_temp).^2)
#        println("ret $ret")
        return constval*ret
    end

    opts = Optim.Options( f_tol = 1e-12, g_tol = 1e-12, iterations = 500000, store_trace = true, show_trace = false)
    ret1 = Optim.optimize(f, [4,3,2,1,0,0,1,2,3,4.0]*-0.2, opts)

    #    MAT = zeros(11,11)
    MAT = zeros(11,11)
    GGG = zeros(11)
    grad_tmp = zeros(10)
    n_temp = zeros(10)
    function grad!(grad_TMP, v)
        grad_TMP .= 0.0
        H .= T + diagm(v)
        vals, vects = eigen(H)
        n_temp .= 0.0
        for i = 1:2
            n_temp += vects[:,i].^2
        end

        for i = 1:2
            #        println("n_temp - n ", n_temp - n)
            MAT[1:10,1:10] = H - I(10)*vals[i]
            MAT[1:10,11] = 2* vects[:,i]
            MAT[11,1:10] = 2* vects[:,i]'
            GGG[1:10] = constval * 4 * (n - n_temp ) .* vects[:,i]
            c = MAT \ GGG
#            println("c $c")
            grad_TMP .+= c[1:10] .* vects[:,i]
        end
#        println("grad TMP $grad_TMP")
#        return grad_TMP
    end

    #=
    println("grad_tmp before ")
    gret = grad!(grad_tmp, [4,3,2,1,0,0,1,2,3,4.0] * 0.2)
    println("grad_tmp $grad_tmp")
    println("gret $gret")

    a = f([4,3,2,1,0,0,1,2,3,4.0] * 0.2)
    b = f([4,3,2,1,0,0,1,2,3,4.0] * 0.2 + [0.01,0,0,0,0,0,0,0,0,0])

    grad1 = (b-a) / 0.01
    println("grad1 $grad1")
    
    a = f([4,3,2,1,0,0,1,2,3,4.0] * 0.2)
    b = f([4,3,2,1,0,0,1,2,3,4.0] * 0.2 + [0.001,0,0,0,0,0,0,0,0,0])

    grad1 = (b-a) / 0.001
    println("grad1 $grad1")

    a = f([4,3,2,1,0,0,1,2,3,4.0] * 0.2)
    b = f([4,3,2,1,0,0,1,2,3,4.0] * 0.2 + [0.0001,0,0,0,0,0,0,0,0,0])

    grad1 = (b-a) / 0.0001
    println("grad1 $grad1")

    a = f([4,3,2,1,0,0,1,2,3,4.0] * 0.2)
    b = f([4,3,2,1,0,0,1,2,3,4.0] * 0.2 + [0.00001,0,0,0,0,0,0,0,0,0])

    grad1 = (b-a) / 0.00001
    println("grad1 $grad1")

    a = f([4,3,2,1,0,0,1,2,3,4.0] * 0.2)
    b = f([4,3,2,1,0,0,1,2,3,4.0] * 0.2 + [0.000001,0,0,0,0,0,0,0,0,0])

    grad1 = (b-a) / 0.000001
    println("grad1 $grad1")
=#    

    opts = Optim.Options( f_tol = 1e-12, g_tol = 1e-12, iterations = 500000, store_trace = true, show_trace = false)
    ret2 = Optim.optimize(f, grad!, [4,3,2,1,0,0,1,2,3,4.0]*0.2, BFGS(), opts)
    return ret1, ret2
#    return missing, missing
end
