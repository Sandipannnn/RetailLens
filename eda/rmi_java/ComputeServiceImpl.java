import java.rmi.RemoteException;
import java.rmi.server.UnicastRemoteObject;
import java.time.LocalTime;

public class ComputeServiceImpl extends UnicastRemoteObject implements ComputeService {

    public ComputeServiceImpl() throws RemoteException {
        super();
    }

    @Override
    public int computeSquare(int number) throws RemoteException {
        System.out.println("[SERVER EVENT] Received RMI request to square: " + number);
        return number * number;
    }

    @Override
    public String fetchNodeTime() throws RemoteException {
        System.out.println("[SERVER EVENT] Received RMI request for clock sync.");
        return "Server Local Time: " + LocalTime.now().toString();
    }
}