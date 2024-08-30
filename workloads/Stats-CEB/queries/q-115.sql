SELECT COUNT(*)
FROM comments AS c,
  postHistory AS ph,
  badges AS b,
  users AS u
WHERE u.Id = b.UserId
  AND u.Id = ph.UserId
  AND u.Id = c.UserId
  AND c.CreationDate <= CAST('2014-08-28 00:18:24' AS timestamp)
  AND b.Date >= CAST('2010-09-15 02:50:48' AS timestamp)
  AND u.Reputation >= 1
  AND u.Reputation <= 1443
  AND u.DownVotes >= 0
  AND u.DownVotes <= 3;
