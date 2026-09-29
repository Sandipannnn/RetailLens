import java.rmi.registry.LocateRegistry;
import java.rmi.registry.Registry;

public class RMIClient {

    public static void main(String[] args) {

        String targetHost = "127.0.0.1";
        int targetPort = 1099;

        try {
            System.out.println("[RMI CLIENT] Locating RMI Registry at "
                    + targetHost + ":" + targetPort);

            Registry registry = LocateRegistry.getRegistry(targetHost, targetPort);

            ComputeService compProxy = (ComputeService) registry.lookup("DistributedComputeEngine");

            System.out.println("[RMI CLIENT] Remote proxy stub resolved successfully.");

            int inputVal = 12;

            System.out.println(
                    "\n[Action] Executing remote call 'computeSquare("
                            + inputVal + ")'...");

            int squareResult = compProxy.computeSquare(inputVal);

            System.out.println("[Outcome] Return response received: " + squareResult);

            System.out.println("\n[Action] Executing remote call 'fetchNodeTime()'...");

            String timeResult = compProxy.fetchNodeTime();

            System.out.println("[Outcome] Return response received: " + timeResult);

        } catch (Exception e) {

            System.err.println(
                    "[RMI CLIENT ERROR] Failed to complete invocation: "
                            + e.getMessage());

            e.printStackTrace();
        }
    }
}