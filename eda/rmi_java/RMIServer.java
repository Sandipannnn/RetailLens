import java.rmi.registry.LocateRegistry;
import java.rmi.registry.Registry;

public class RMIServer {

    public static void main(String[] args) {

        try {
            Registry registry = LocateRegistry.createRegistry(1099);

            System.out.println("[RMI REGISTRY] Created local naming service container on port 1099.");

            ComputeService serviceInstance = new ComputeServiceImpl();

            registry.rebind("DistributedComputeEngine", serviceInstance);

            System.out.println("[RMI SERVER READY] Service 'DistributedComputeEngine' safely bound.");

        } catch (Exception e) {
            System.err.println("[RMI SERVER FAILURE] Abrupt initialization crash: "
                    + e.getMessage());

            e.printStackTrace();
        }
    }
}