SELECT COUNT(*)
FROM comments AS c,
  postHistory AS ph,
  votes AS v,
  users AS u
WHERE u.Id = v.UserId
  AND v.UserId = ph.UserId
  AND ph.UserId = c.UserId
  AND v.BountyAmount >= 0
  AND v.CreationDate >= CAST('2010-07-26 00:00:00' AS timestamp)
  AND v.CreationDate <= CAST('2014-09-08 00:00:00' AS timestamp)
  AND u.Reputation >= 1
  AND u.Views >= 0
  AND u.Views <= 110
  AND u.UpVotes = 0
  AND u.CreationDate >= CAST('2010-07-28 19:29:11' AS timestamp)
  AND u.CreationDate <= CAST('2014-08-14 05:29:30' AS timestamp);
