@Test
public void testFlakySample() throws Exception {
    Thread.sleep(2000);
    Random r = new Random();
    int x = r.nextInt();
}